#!/usr/bin/env python3
"""Verify -que words against Wiktionary to identify false positives."""

import time
import ssl
import urllib.request
import urllib.error
from pathlib import Path

# Create SSL context that doesn't verify certificates
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def check_wiktionary(word: str, max_retries: int = 3) -> bool:
    """Check if a word exists in Wiktionary Latin section with retries."""
    url = f"https://en.wiktionary.org/wiki/{word}"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(
                req, timeout=10, context=ssl_context
            ) as response:
                content = response.read().decode("utf-8")
                # Check if there's a Latin section - Wiktionary uses various formats
                # Look for Latin heading markers in the HTML
                # The page has "## Latin" or "==Latin==" or id="Latin" or id="la"
                return "Latin" in content and (
                    "## Latin" in content
                    or "==Latin==" in content
                    or 'id="Latin"' in content
                    or 'id="la"' in content
                    or 'lang="la"' in content
                )
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} for {word}...")
                time.sleep(2)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} for {word} (error: {e})...")
                time.sleep(2)
                continue
            print(f"Error checking {word}: {e}")
            return False

    return False


def get_base_word(word: str) -> str:
    """Get the base word by removing -que suffix."""
    if word.endswith("que"):
        return word[:-3]
    return word


def main():
    blacklist_file = Path(
        "/Users/ben/code/latin-masking/src/latin_masking/data/que_blacklist.txt"
    )

    with open(blacklist_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Separate already-marked words from unmarked ones
    already_verified = []  # Words already marked as verified
    already_base_found = []  # Words already marked as ?!
    already_not_found = []  # Words already marked as ??
    words_to_check = []  # Unmarked words to check

    for line in lines:
        if line.startswith("??"):
            already_not_found.append(line)
        elif line.startswith("?!"):
            already_base_found.append(line)
        elif line.startswith("?"):
            # Unknown marker, treat as unmarked
            words_to_check.append(line.lstrip("?!"))
        else:
            words_to_check.append(line)

    verified_words = []  # Words found in Wiktionary
    base_found_words = []  # Words not found but base word is found (marked ?!)
    not_found_words = []  # Neither word nor base found (marked ??)

    print(f"Checking {len(words_to_check)} words against Wiktionary...")
    print(
        f"(Preserving {len(already_verified)} verified, {len(already_base_found)} ?!, {len(already_not_found)} ??)"
    )

    for i, word in enumerate(words_to_check):
        base_word = get_base_word(word)
        print(
            f"[{i+1}/{len(words_to_check)}] Checking {word} (base: {base_word})...",
            end=" ",
        )

        if check_wiktionary(word):
            print("✓ Found")
            verified_words.append(word)
        elif check_wiktionary(base_word):
            print("✓ Base found")
            base_found_words.append(word)
        else:
            print("✗ Neither found")
            not_found_words.append(word)

        # Rate limit: 1 request per second
        time.sleep(1)

    print(f"\n=== Results ===")
    print(f"Verified (word in Wiktionary): {len(verified_words)}")
    print(f"Base found (base word in Wiktionary): {len(base_found_words)}")
    print(f"Not found (neither in Wiktionary): {len(not_found_words)}")

    # Write all words back to blacklist with markers
    with open(blacklist_file, "w", encoding="utf-8") as f:
        for word in verified_words:
            f.write(f"{word}\n")
        for word in base_found_words:
            f.write(f"?!{word}\n")
        for word in not_found_words:
            f.write(f"??{word}\n")
        # Preserve already-marked words
        for word in already_verified:
            f.write(f"{word}\n")
        for word in already_base_found:
            f.write(f"{word}\n")
        for word in already_not_found:
            f.write(f"{word}\n")

    print(
        f"\nUpdated blacklist with {len(verified_words)} verified, "
        f"{len(base_found_words)} base-found (?!), "
        f"{len(not_found_words)} not-found (??)"
    )


if __name__ == "__main__":
    main()
