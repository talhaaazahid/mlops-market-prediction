"""Reddit finance data ingestion using PRAW.

Scrapes hot + new posts from 5 finance subreddits, extracts top comments,
maps ticker mentions, and appends to data/raw/reddit_posts.csv.

Required env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import praw
from praw.exceptions import PRAWException
from prawcore.exceptions import PrawcoreException

from src.config import REDDIT_FILE, REDDIT_SUBREDDITS, get_env, get_params
from src.utils.helpers import (
    append_csv,
    deduplicate_dataframe,
    extract_ticker_mentions,
    to_utc,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def _build_reddit_client() -> praw.Reddit:
    """Instantiate a read-only PRAW Reddit client from env vars.

    Returns:
        Authenticated (read-only) PRAW Reddit instance.

    Raises:
        RuntimeError: If required credentials are absent.
    """
    client_id = get_env("REDDIT_CLIENT_ID", required=True)
    client_secret = get_env("REDDIT_CLIENT_SECRET", required=True)
    user_agent = get_env("REDDIT_USER_AGENT", required=True)

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=300,
    )


def _extract_post(submission: Any, subreddit_name: str, top_n_comments: int = 5) -> dict[str, Any]:
    """Convert a PRAW Submission to a flat dict.

    Args:
        submission: A `praw.models.Submission` object.
        subreddit_name: Subreddit name string.
        top_n_comments: Max number of top-score comments to include.

    Returns:
        Dict with keys matching the `reddit_posts.csv` schema.
    """
    title = submission.title or ""
    selftext = (submission.selftext or "").strip()

    # Fetch top comments (sorted by score, skip deleted/empty)
    top_comments: list[str] = []
    try:
        submission.comments.replace_more(limit=0)
        sorted_comments = sorted(
            submission.comments.list(),
            key=lambda c: getattr(c, "score", 0),
            reverse=True,
        )
        for comment in sorted_comments[:top_n_comments]:
            body = getattr(comment, "body", "")
            if body and body not in ("[deleted]", "[removed]"):
                top_comments.append(body[:500])
    except (PRAWException, PrawcoreException) as exc:
        log.debug("Could not load comments for %s: %s", submission.id, exc)

    full_text = " ".join(filter(None, [title, selftext, *top_comments]))
    ticker_mentions = extract_ticker_mentions(full_text)

    created_utc = to_utc(submission.created_utc)
    published_date = created_utc.isoformat() if created_utc else None

    return {
        "post_id": submission.id,
        "subreddit": subreddit_name,
        "title": title,
        "selftext": selftext,
        "score": int(submission.score),
        "num_comments": int(submission.num_comments),
        "created_utc": published_date,
        "permalink": f"https://www.reddit.com{submission.permalink}",
        "top_comments": json.dumps(top_comments),
        "full_text": full_text,
        "ticker_mentions": json.dumps(ticker_mentions),
    }


def fetch_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int = 50,
    min_score: int = 5,
    top_n_comments: int = 5,
) -> list[dict[str, Any]]:
    """Scrape hot + new listings from one subreddit.

    Args:
        reddit: Authenticated PRAW client.
        subreddit_name: Subreddit name (without `r/`).
        limit: Max posts per listing (hot and new separately).
        min_score: Drop posts with score below this threshold.
        top_n_comments: Top comments per post.

    Returns:
        List of post dicts.
    """
    posts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    sub = reddit.subreddit(subreddit_name)

    for listing_name, listing in (("hot", sub.hot(limit=limit)), ("new", sub.new(limit=limit))):
        try:
            for submission in listing:
                if submission.id in seen_ids:
                    continue
                if submission.score < min_score:
                    continue
                seen_ids.add(submission.id)
                try:
                    post = _extract_post(submission, subreddit_name, top_n_comments)
                    posts.append(post)
                except Exception as exc:
                    log.debug("Skipping post %s: %s", submission.id, exc)
        except (PRAWException, PrawcoreException) as exc:
            log.error("Error reading r/%s %s: %s", subreddit_name, listing_name, exc)

    log.info("r/%s: %d posts collected", subreddit_name, len(posts))
    return posts


def fetch_all_reddit(
    subreddits: dict[str, int] | None = None,
    top_n_comments: int | None = None,
    min_score: int | None = None,
) -> pd.DataFrame:
    """Scrape all configured subreddits.

    Args:
        subreddits: Mapping of {subreddit_name: post_limit}.
            Defaults to `config.REDDIT_SUBREDDITS`.
        top_n_comments: Comments per post.
        min_score: Minimum score threshold.

    Returns:
        Deduplicated DataFrame of Reddit posts.
    """
    params = get_params()
    if subreddits is None:
        subreddits = REDDIT_SUBREDDITS
    if top_n_comments is None:
        top_n_comments = int(params["ingestion"].get("reddit_top_comments", 5))
    if min_score is None:
        min_score = int(params["ingestion"].get("min_reddit_score", 5))

    try:
        reddit = _build_reddit_client()
    except RuntimeError as exc:
        log.error("Cannot build Reddit client — skipping: %s", exc)
        return pd.DataFrame()

    all_posts: list[dict[str, Any]] = []
    for sub_name, limit in subreddits.items():
        posts = fetch_subreddit(
            reddit,
            sub_name,
            limit=limit,
            min_score=min_score,
            top_n_comments=top_n_comments,
        )
        all_posts.extend(posts)

    if not all_posts:
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    df = deduplicate_dataframe(df, key_col="post_id")
    log.info("Reddit: total %d posts after dedup", len(df))
    return df


def fetch_latest_reddit(save: bool = True) -> pd.DataFrame:
    """Convenience wrapper for pipeline automation.

    Args:
        save: Persist new posts to disk when True.

    Returns:
        DataFrame of newly fetched Reddit posts.
    """
    df = fetch_all_reddit()
    if save and not df.empty:
        append_csv(df, REDDIT_FILE, dedup_key="post_id")
    return df


if __name__ == "__main__":
    posts = fetch_latest_reddit(save=True)
    if not posts.empty:
        print(posts[["subreddit", "title", "score", "ticker_mentions"]].head(10).to_string())
    else:
        print("No posts fetched (check Reddit API credentials in .env)")
