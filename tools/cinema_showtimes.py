"""Four read-only movie tools over Maoyan's public schedule responses."""

from __future__ import annotations

import difflib
import html
import io
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from langchain.tools import tool


BASE = "https://m.maoyan.com"
BEIJING = timezone(timedelta(hours=8))
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": BASE + "/"}


def _bytes(url: str, limit: int = 2_000_000) -> bytes:
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=12) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError("猫眼返回内容超过大小上限。")
            return body
    except HTTPError as exc:
        raise ValueError(f"猫眼返回 HTTP {exc.code}，本次无法确认排片。") from None
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"猫眼连接失败（{type(exc).__name__}）。") from None


def _json(path: str, params: dict | None = None) -> dict:
    url = BASE + path + ("?" + urlencode(params) if params else "")
    try:
        data = json.loads(_bytes(url))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("猫眼没有返回可读取的 JSON 排片数据。") from None
    if not isinstance(data, dict):
        raise ValueError("猫眼排片数据格式无法识别。")
    return data


def _day(value: str) -> date:
    today = datetime.now(BEIJING).date()
    if value.strip() in {"today", "今天"}:
        return today
    if value.strip() in {"tomorrow", "明天"}:
        return today + timedelta(days=1)
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError("date 请用 today、tomorrow 或 YYYY-MM-DD。") from None


def _city(city: str) -> tuple[int, str]:
    if not city.strip():
        raise ValueError("请指定城市或县市，例如莱西。")
    cities = _json("/dianying/cities.json").get("cts")
    if not isinstance(cities, list):
        raise ValueError("猫眼城市清单不可用。")
    wanted = city.strip().removesuffix("市")
    matches = [item for item in cities if isinstance(item, dict)
               and str(item.get("nm", "")).removesuffix("市") == wanted]
    if len(matches) != 1:
        raise ValueError("无法在猫眼城市清单中唯一确定该城市，请填写城市或县市全名。")
    return int(matches[0]["id"]), str(matches[0]["nm"])


def _genres(description: str) -> list[str]:
    parts = [part.strip() for part in description.split("|")]
    return [part.strip() for part in parts[1].replace("/", ",").split(",") if part.strip()] if len(parts) > 1 else []


def _render_digit(font, char: str):
    from PIL import Image, ImageDraw
    image = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(image).text((5, -5), char, font=font, fill=255)
    box = image.getbbox()
    return image.crop(box).resize((32, 48)) if box else image


@lru_cache(maxsize=16)
def _font_digits(url: str) -> dict[int, str]:
    """Match this response's private glyphs against locally installed digit outlines."""
    if urlsplit(url).hostname != "s3plus.meituan.net" or not url.endswith(".woff"):
        return {}
    try:
        from fontTools.ttLib import TTFont
        from PIL import ImageChops, ImageFont, ImageStat
        font = TTFont(io.BytesIO(_bytes(url, 300_000)))
        codepoints = [point for point in font.getBestCmap() if 0xE000 <= point <= 0xF8FF]
        if len(codepoints) != 10:
            return {}
        font.flavor = None
        output = io.BytesIO()
        font.save(output)
        output.seek(0)
        coded = ImageFont.truetype(output, 70)
        reference = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 70)
        samples = [_render_digit(reference, str(digit)) for digit in range(10)]
        mapping = {}
        for point in codepoints:
            sample = _render_digit(coded, chr(point))
            scores = sorted((ImageStat.Stat(ImageChops.difference(sample, ref)).sum[0], digit)
                            for digit, ref in enumerate(samples))
            if scores[0][0] * 1.5 >= scores[1][0]:
                return {}
            mapping[point] = str(scores[0][1])
        return mapping if len(set(mapping.values())) == 10 else {}
    except (OSError, ValueError, ImportError):
        return {}


def _price(raw: str, css: str) -> float | None:
    match = re.search(r'url\("(//[^"?]+\.woff)"\)', css)
    mapping = _font_digits("https:" + match.group(1)) if match else {}
    plain = html.unescape(re.sub(r"<[^>]*>", "", raw))
    decoded = "".join(mapping.get(ord(char), "?") if 0xE000 <= ord(char) <= 0xF8FF else char
                      for char in plain).replace("¥", "").replace("￥", "").strip()
    if "?" in decoded or not re.fullmatch(r"\d{1,3}(?:\.\d{1,2})?", decoded):
        return None
    value = float(decoded)
    return value if 0 < value < 1000 else None


def _listing(city: str) -> dict:
    city_id, city_name = _city(city)
    entries, total, has_more = [], 0, True
    while has_more and len(entries) < 200:
        listing = _json("/ajax/cinemaList", {"ci": city_id, "offset": len(entries), "limit": 50})
        batch, paging = listing.get("cinemas"), listing.get("paging")
        if not isinstance(batch, list) or not isinstance(paging, dict):
            raise ValueError("猫眼影院列表格式无法识别。")
        total = int(paging.get("total", len(entries) + len(batch)))
        entries.extend(batch)
        has_more = bool(paging.get("hasMore")) and bool(batch)
    return {"city": city_name, "entries": entries, "total": total,
            "complete": not has_more and len(entries) >= total}


def _choose(query: str, names: list[str]) -> tuple[str | None, list[str]]:
    wanted = query.strip().casefold()
    if not wanted:
        return None, names[:5]
    exact = [name for name in names if name.casefold() == wanted]
    if len(exact) == 1:
        return exact[0], []
    partial = [name for name in names if wanted in name.casefold()]
    if len(partial) == 1:
        return partial[0], []
    return None, (partial or difflib.get_close_matches(query.strip(), names, n=5))[:10]


def list_cinemas(city: str) -> dict:
    """List Maoyan-listed local cinemas and exact names for follow-up schedule queries."""
    try:
        listing = _listing(city)
        return {"source": "maoyan_public_schedule", "city": listing["city"],
                "checked_at": datetime.now(BEIJING).isoformat(timespec="seconds"),
                "listed_cinemas": listing["total"], "complete": listing["complete"],
                "cinemas": [{"cinema": str(item.get("nm") or ""), "cinema_id": item["id"],
                             "address": str(item.get("addr") or ""),
                             "url": f"https://www.maoyan.com/cinema/{item['id']}"}
                            for item in listing["entries"] if isinstance(item, dict) and isinstance(item.get("id"), int)]}
    except ValueError as exc:
        return {"error": str(exc), "source": "maoyan_public_schedule"}


def _schedule(city: str, date_value: str, *, prices: bool, cinema: str = "") -> dict:
    day = _day(date_value)
    listing = _listing(city)
    entries, total, complete = listing["entries"], listing["total"], listing["complete"]
    matched_cinema = None
    if cinema:
        names = [str(item.get("nm") or "") for item in entries if isinstance(item, dict)]
        matched_cinema, suggestions = _choose(cinema, names)
        if not complete and matched_cinema and matched_cinema.casefold() != cinema.strip().casefold():
            matched_cinema = None
        if matched_cinema is None:
            return {"error": f"无法唯一确定影院“{cinema}”，请从候选影院中选择。",
                    "suggestions": suggestions, "available_cinemas": names,
                    "source": "maoyan_public_schedule", "city": listing["city"],
                    "date": day.isoformat(), "complete": complete}
        entries = [item for item in entries if isinstance(item, dict) and item.get("nm") == matched_cinema]
    elif len(entries) > 50:
        entries, complete = entries[:50], False
    cinemas, errors, date_listed = [], [], False
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            complete = False
            continue
        cinema_id, name = item["id"], str(item.get("nm", ""))
        try:
            detail = _json("/ajax/cinemaDetail", {"cinemaId": cinema_id})
            show_data = detail.get("showData")
            if not isinstance(show_data, dict) or not isinstance(show_data.get("movies"), list):
                raise ValueError("没有返回可核验的影片清单。")
        except ValueError as exc:
            complete = False
            errors.append({"cinema": name, "error": str(exc)})
            continue
        css = str((detail.get("stone") or {}).get("cssSource", "")) if prices else ""
        films = []
        for film in show_data["movies"]:
            if not isinstance(film, dict):
                continue
            date_listed |= any(isinstance(group, dict) and group.get("showDate") == day.isoformat()
                               for group in film.get("shows", []))
            sessions = []
            for group in film.get("shows", []):
                if not isinstance(group, dict) or group.get("showDate") != day.isoformat():
                    continue
                for show in group.get("plist", []):
                    if not isinstance(show, dict) or show.get("dt") != day.isoformat():
                        continue
                    start = str(show.get("tm", ""))
                    try:
                        started = datetime.strptime(f"{day} {start}", "%Y-%m-%d %H:%M")
                    except ValueError:
                        continue
                    duration = film.get("dur")
                    estimated_end = None
                    if isinstance(duration, int) and 1 <= duration <= 360:
                        estimated_end = (started + timedelta(minutes=duration)).strftime("%Y-%m-%d %H:%M")
                    member = str(show.get("vipPrice") or "")
                    sessions.append({
                        "start_time": start, "end_time_estimate": estimated_end,
                        "hall": str(show.get("th") or ""), "format": str(show.get("tp") or ""),
                        "language": str(show.get("lang") or ""),
                        "price_cny": _price(str(show.get("sellPr") or ""), css) if prices else None,
                        "member_price_cny": float(member) if re.fullmatch(r"\d+(?:\.\d+)?", member) else None,
                        "member_price_name": str(show.get("vipPriceName") or ""),
                    })
            if sessions:
                films.append({"movie": str(film.get("nm") or ""), "movie_id": film.get("id"),
                              "genres": _genres(str(film.get("desc") or "")),
                              "shows": sorted(sessions, key=lambda s: s["start_time"])})
        cinemas.append({"cinema": name, "cinema_id": cinema_id,
                        "address": str(item.get("addr") or ""),
                        "url": f"https://www.maoyan.com/cinema/{cinema_id}", "movies": films})
    return {"source": "maoyan_public_schedule", "city": listing["city"], "date": day.isoformat(),
            "checked_at": datetime.now(BEIJING).isoformat(timespec="seconds"),
            "coverage": ("指定影院" if cinema else
                         "猫眼当前列出的前50家影院" if total > 50 else "猫眼当前列出的影院"),
            "listed_cinemas": total, "matched_cinema": matched_cinema,
            "checked_cinemas": len(cinemas),
            "complete_for_listed_cinemas": complete and not bool(cinema),
            "complete_for_requested_scope": complete,
            "date_listed_in_source": date_listed,
            "cinemas": cinemas, "errors": errors}


def list_city_movies(city: str, date: str = "today", genre: str = "") -> dict:
    """查询某城市在指定日期实际有排片的去重影片清单；城市级问题只调用本工具一次。

    用于“莱西明天有什么电影”“某城市今晚有什么爱情片/动作片”等按城市、日期和
    可选类型查询。工具内部已经遍历猫眼当前列出的当地影院并聚合影片；不要先调用
    list_cinemas，再逐家调用 get_cinema_showtimes。只有用户明确指定某一家影院时才
    使用 get_cinema_showtimes；指定某一部影片并询问具体场次或票价时使用
    get_movie_showtimes。genre 可填“爱情”“动作”“恐怖”等；本工具不负责评分或影评。
    """
    try:
        data = _schedule(city, date, prices=False)
        wanted_genre = genre.strip().removesuffix("片")
        movies = {}
        for cinema in data.pop("cinemas"):
            for film in cinema["movies"]:
                if wanted_genre and not any(wanted_genre in category for category in film["genres"]):
                    continue
                movie = movies.setdefault(film["movie"], {"movie": film["movie"],
                    "genres": film["genres"], "cinemas": [], "show_count": 0})
                movie["cinemas"].append(cinema["cinema"])
                movie["show_count"] += len(film["shows"])
        data["movies"] = sorted(movies.values(), key=lambda film: film["movie"])
        data["genre_filter"] = wanted_genre or None
        if not data["movies"]:
            data["message"] = ("猫眼当前列出的影院在该日期没有匹配的影片；请检查类型或日期。"
                               if data["complete_for_listed_cinemas"] and data["date_listed_in_source"]
                               else "指定日期的排片可能尚未发布，或部分影院未核验，暂无可确认的匹配影片。")
        return data
    except ValueError as exc:
        return {"error": str(exc), "source": "maoyan_public_schedule"}


def get_cinema_showtimes(city: str, cinema: str, date: str = "today") -> dict:
    """List all dated movies, sessions and public prices at one cinema; unique partial names work."""
    try:
        if not cinema.strip():
            raise ValueError("请提供影院名称；可先调用 list_cinemas 查看当地影院。")
        data = _schedule(city, date, prices=True, cinema=cinema)
        if "error" in data:
            return data
        data["price_note"] = "price_cny 为公开场次标价；null 表示动态字体未能可靠解码。会员价可能有资格限制。"
        data["unsupported_fields"] = ["remaining_seats", "occupancy", "booking_count"]
        if data["cinemas"] and not data["cinemas"][0]["movies"]:
            data["message"] = "该影院在指定日期暂无可确认的场次；排片也可能尚未发布。"
        return data
    except ValueError as exc:
        return {"error": str(exc), "source": "maoyan_public_schedule"}


def get_movie_showtimes(city: str, movie: str, date: str = "today", after: str = "") -> dict:
    """Find a full or unique partial movie title across city cinemas, with sessions and prices."""
    try:
        if not movie.strip():
            raise ValueError("请提供影片名称；可先调用 list_city_movies 查看当天影片。")
        if after:
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", after):
                raise ValueError("after 请用 HH:MM，例如 19:00。")
        data = _schedule(city, date, prices=True)
        titles = sorted({film["movie"] for cinema in data["cinemas"] for film in cinema["movies"]})
        found, hints = _choose(movie, titles)
        if found is None:
            reason = ("片名有多个匹配，请从候选影片中选择。" if hints
                      else "没有匹配的当日排片；片名可能有误，或排片尚未发布。")
            return {"error": f"{data['date']} 《{movie.strip()}》：{reason}",
                    "suggestions": hints, "available_movies": titles,
                    "source": "maoyan_public_schedule",
                    "complete_for_listed_cinemas": data["complete_for_listed_cinemas"],
                    "date_listed_in_source": data["date_listed_in_source"],
                    "city": data["city"], "date": data["date"], "errors": data["errors"]}
        for cinema in data["cinemas"]:
            cinema["movies"] = [film for film in cinema["movies"] if film["movie"] == found]
            for film in cinema["movies"]:
                film["shows"] = [show for show in film["shows"] if not after or show["start_time"] >= after]
        data["cinemas"] = [cinema for cinema in data["cinemas"] if any(film["shows"] for film in cinema["movies"])]
        data["movie"] = found
        data["name_match"] = "exact" if found.casefold() == movie.strip().casefold() else "unique_partial"
        data["price_note"] = "price_cny 是对应场次的公开标价；null 表示动态字体未能可靠解码。会员价单列且可能有资格限制。"
        data["end_time_note"] = "end_time_estimate 为开场加片长的估算，不含映前广告。"
        data["unsupported_fields"] = ["remaining_seats", "occupancy", "booking_count"]
        if not data["cinemas"]:
            data["message"] = "该影片当天有排片，但指定时段后没有可确认的场次。"
        return data
    except ValueError as exc:
        return {"error": str(exc), "source": "maoyan_public_schedule"}


list_cinemas_tool = tool("list_cinemas")(list_cinemas)
list_city_movies_tool = tool("list_city_movies")(list_city_movies)
get_cinema_showtimes_tool = tool("get_cinema_showtimes")(get_cinema_showtimes)
get_movie_showtimes_tool = tool("get_movie_showtimes")(get_movie_showtimes)
