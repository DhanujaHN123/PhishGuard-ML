
import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "verification",
    "account", "secure", "security", "update",
    "password", "credential", "authenticate",
    "bank", "banking", "payment", "billing",
    "wallet", "card", "transaction", "urgent",
    "suspended", "blocked", "expired"
}


def extract_features(url):
    url = str(url).strip()
    parsed = urlparse(
        url if "://" in url else "//" + url
    )

    domain = (parsed.hostname or "").lower()
    url_lower = url.lower()

    try:
        ipaddress.ip_address(domain)
        has_ip = 1
    except ValueError:
        has_ip = 0

    if has_ip:
        subdomain_count = 0
    else:
        subdomain_count = max(0, len(domain.split(".")) - 2)

    tokens = re.split(r"[^a-zA-Z0-9]+", url_lower)
    keyword_count = sum(
        token in SUSPICIOUS_KEYWORDS for token in tokens
    )

    digit_count = sum(char.isdigit() for char in url)
    special_count = sum(
        not char.isalnum() for char in url
    )

    char_counts = Counter(url)
    entropy = 0.0

    if url:
        for count in char_counts.values():
            probability = count / len(url)
            entropy -= probability * math.log2(probability)

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "subdomain_count": subdomain_count,
        "digit_count": digit_count,
        "digit_ratio": digit_count / max(len(url), 1),
        "special_count": special_count,
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "query_parameter_count": (
            len(parsed.query.split("&"))
            if parsed.query else 0
        ),
        "dot_count": url.count("."),
        "path_segments": len([
            part for part in parsed.path.split("/") if part
        ]),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "has_suspicious_keyword": int(keyword_count > 0),
        "keyword_count": keyword_count,
        "entropy": entropy,
        "has_punycode": int("xn--" in domain),
        "has_fragment": int(bool(parsed.fragment))
    }

    return features


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://paypal-login-security.com/verify/account",
        "http://192.168.1.100/login",
        "https://xn--pple-43d.com/login"
    ]

    for url in test_urls:
        print("\nURL:", url)
        for feature, value in extract_features(url).items():
            print(f"{feature}: {value}")
