# Redis Vector Database PoC

This is a proof of concept for using Redis as a vector database.

## Prerequisites

This application assumes that you have a RedisSearch server running on `localhost:6379`. The easiest way to do this is to use Docker:

```bash
docker run -p 6379:6379 redislabs/redisearch:latest
```

Additionally, you'll need an API token from Huggingface, which you can generate [here](https://huggingface.co/settings/tokens).

Once you have your token, copy the `template.env` file to `.env` and paste your token into the `HG_TOKEN` variable.

```shell
cp template.env .env
```

## Running the application

```shell
go run . --load --query=<SEARCH TERM OR PHRASE>
```
