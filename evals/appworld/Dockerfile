FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APPWORLD_ROOT=/world HOME=/tmp
ARG APPWORLD_ALLOWED_SPLITS=train,dev
ENV PERSONALOPS_APPWORLD_ALLOWED_SPLITS=${APPWORLD_ALLOWED_SPLITS}
RUN python -m pip install --no-cache-dir appworld==0.1.3.post1 && python -m appworld.cli install
WORKDIR /world
# Retain the completed bootstrap layer, then enforce all recorded dependency versions.
COPY requirements.snapshot.txt /opt/requirements.lock
RUN python -m pip install --no-cache-dir -r /opt/requirements.lock && python -m pip check
COPY data-0.1.0.bundle /opt/data-0.1.0.bundle
COPY bootstrap_data.py /opt/bootstrap_data.py
RUN python /opt/bootstrap_data.py && mkdir -p /world/experiments && chmod 755 /world
COPY worker.py /opt/personalops_worker.py
USER 10001:10001
ENTRYPOINT ["python", "-u", "/opt/personalops_worker.py"]
