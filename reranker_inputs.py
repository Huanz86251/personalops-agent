"""Cross-encoder sequence capacity, measured with its own tokenizer."""


def effective_limit(model, requested):
    config = model.model.config
    limits = [int(requested)]
    positions = getattr(config, 'max_position_embeddings', None)
    if isinstance(positions, int) and positions > 0:
        # RoBERTa position IDs start at padding_idx + 1, including special tokens.
        if getattr(config, 'model_type', '') in {'roberta', 'xlm-roberta', 'camembert', 'longformer'}:
            positions -= int(getattr(config, 'pad_token_id', 1)) + 1
        limits.append(positions)
    tokenizer_limit = getattr(model.tokenizer, 'model_max_length', None)
    if isinstance(tokenizer_limit,int) and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    limit = min(limits)
    if limit < model.tokenizer.num_special_tokens_to_add(pair=True)+2:
        raise ValueError('Reranker sequence capacity is too small for a query/document pair')
    return limit


def fit_query_head_tail(model, query, documents, limit):
    """Fit a query into every query/document pair while retaining both ends."""
    docs = list(documents)
    if not docs:
        return query
    tokenizer = model.tokenizer
    special = tokenizer.num_special_tokens_to_add(pair=True)
    usable = limit - special
    if usable < 2:
        raise ValueError('Reranker sequence capacity is too small for a query/document pair')
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    document_lengths = [
        len(tokenizer.encode(document, add_special_tokens=False))
        for document in docs
    ]
    # Keep at most half the pair for the longest capability card; the remainder
    # belongs to the query. Short cards naturally leave the same bounded query.
    document_reserve = min(max(document_lengths, default=1), max(1, usable // 2))
    query_capacity = max(1, usable - document_reserve)
    if len(query_ids) <= query_capacity and all(
        len(tokenizer(query, document, truncation=False, verbose=False)['input_ids']) <= limit
        for document in docs
    ):
        return query

    kept = min(len(query_ids), query_capacity)
    while kept > 0:
        head = (kept + 1) // 2
        tail = kept // 2
        fitted = tokenizer.decode(
            query_ids[:head] + (query_ids[-tail:] if tail else []),
            skip_special_tokens=True,
        )
        if all(
            len(tokenizer(fitted, document, truncation=False, verbose=False)['input_ids']) <= limit
            for document in docs
        ):
            return fitted
        kept -= 1
    raise ValueError('No query token fits reranker pairs')


def describe_pairs(model, pairs, limit):
    records=[]
    for i,(query,document) in enumerate(pairs):
        before=model.tokenizer(query,document,truncation=False,verbose=False)['input_ids']
        after=model.tokenizer(query,document,truncation='longest_first',max_length=limit,verbose=False)['input_ids']
        if len(after)>limit:raise ValueError('Reranker tokenizer did not enforce sequence limit')
        records.append({'index':i,'tokens_before':len(before),'tokens_after':len(after),
                        'truncated':len(before)>len(after),
                        'effective_pair_text':model.tokenizer.decode(after,skip_special_tokens=False)})
    return records


def document_windows(model, query, documents, limit, max_windows=2):
    """Bound each physical pair with the reranker's tokenizer, preserving query."""
    if max_windows < 1:
        raise ValueError('max_windows must be positive')
    t = model.tokenizer
    special = t.num_special_tokens_to_add(pair=True)
    qids = t.encode(query, add_special_tokens=False)
    q = t.decode(qids[:max(1, (limit-special)//2)], skip_special_tokens=True) if len(qids) > (limit-special)//2 else query
    capacity = limit-special-len(t.encode(q, add_special_tokens=False))
    windows, owners, audit = [], [], []
    for index, text in enumerate(documents):
        ids = t.encode(text, add_special_tokens=False)
        start = 0
        for window in range(max_windows):
            end = min(start+capacity, len(ids))
            part = text if start == 0 and end == len(ids) else t.decode(ids[start:end], skip_special_tokens=True)
            # Decode/re-encode may change token count; measure actual pair.
            while len(t(q, part, truncation=False, verbose=False)['input_ids']) > limit and end > start:
                end -= 1
                part = t.decode(ids[start:end], skip_special_tokens=True)
            if end == start and ids:
                raise ValueError('No document token fits reranker pair')
            windows.append(part); owners.append(index)
            audit.append({'document_index': index, 'window': window+1, 'start_token': start,
                          'end_token': end, 'document_tokens': len(ids), 'remaining_tokens': len(ids)-end})
            start = end
            if start >= len(ids): break
    return q, windows, owners, audit
