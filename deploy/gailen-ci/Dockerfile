# Imagen base: el binario de esta rama sobre bookworm-slim, sin punto de entrada propio.
FROM debian:bookworm-slim
LABEL org.opencontainers.image.source="https://github.com/GailenTech/librefang"
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ca-certificates curl jq libdbus-1-3 > /dev/null && rm -rf /var/lib/apt/lists/*
COPY librefang-bin /usr/local/bin/librefang
RUN chmod 0755 /usr/local/bin/librefang && librefang --version
ENV LIBREFANG_HOME=/data
VOLUME /data
EXPOSE 4545
ENTRYPOINT ["librefang"]
