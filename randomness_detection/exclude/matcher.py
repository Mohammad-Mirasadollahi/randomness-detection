"""In-memory wildcard matchers optimized for large suffix/prefix sets."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .normalize import normalize_domain_pattern, normalize_text, reverse_domain


@dataclass(frozen=True)
class WildcardRule:
    pattern: str
    rule_type: str
    normalized: str


class SuffixTrie:
    """Reversed-domain trie for fast *.domain.com matching."""

    def __init__(self) -> None:
        self._root: dict[str, object] = {}

    def add(self, suffix: str, pattern: str) -> None:
        node = self._root
        for label in reverse_domain(suffix).split("."):
            child = node.get(label)
            if not isinstance(child, dict):
                child = {}
                node[label] = child
            node = child
        node["$"] = pattern

    def match(self, domain: str) -> str | None:
        node: object = self._root
        matched_pattern: str | None = None
        for label in reverse_domain(domain).split("."):
            if not isinstance(node, dict):
                break
            if "$" in node:
                matched_pattern = str(node["$"])
            child = node.get(label)
            if not isinstance(child, dict):
                return matched_pattern
            node = child
        if isinstance(node, dict) and "$" in node:
            return str(node["$"])
        return matched_pattern


class PrefixMatcher:
    def __init__(self) -> None:
        self._rules: list[tuple[str, str]] = []

    def add(self, prefix: str, pattern: str) -> None:
        self._rules.append((prefix, pattern))
        self._rules.sort(key=lambda item: len(item[0]), reverse=True)

    def match(self, text: str) -> str | None:
        for prefix, pattern in self._rules:
            if text.startswith(prefix):
                return pattern
        return None


class GlobMatcher:
    def __init__(self) -> None:
        self._rules: list[tuple[str, str]] = []

    def add(self, glob_pattern: str, pattern: str) -> None:
        self._rules.append((glob_pattern, pattern))

    def match(self, text: str) -> str | None:
        for glob_pattern, pattern in self._rules:
            if fnmatch.fnmatchcase(text, glob_pattern):
                return pattern
        return None


class WildcardIndex:
    def __init__(self) -> None:
        self.suffix_trie = SuffixTrie()
        self.prefix_matcher = PrefixMatcher()
        self.glob_matcher = GlobMatcher()
        self.domain_exact: set[str] = set()
        self.domain_suffix: set[str] = set()
        self.rule_count = 0

    def rebuild(self, rules: list[WildcardRule]) -> None:
        self.suffix_trie = SuffixTrie()
        self.prefix_matcher = PrefixMatcher()
        self.glob_matcher = GlobMatcher()
        self.domain_exact = set()
        self.domain_suffix = set()
        self.rule_count = 0

        for rule in rules:
            self.rule_count += 1
            if rule.rule_type == "domain":
                self.domain_exact.add(rule.normalized)
                self.suffix_trie.add(rule.normalized, rule.pattern)
            elif rule.rule_type == "suffix":
                self.domain_suffix.add(rule.normalized)
                self.suffix_trie.add(rule.normalized, rule.pattern)
            elif rule.rule_type == "prefix":
                self.prefix_matcher.add(rule.normalized, rule.pattern)
            elif rule.rule_type == "glob":
                self.glob_matcher.add(rule.normalized, rule.pattern)

    def match_text(self, text: str, domain: str | None) -> str | None:
        if domain:
            if domain in self.domain_exact:
                return f"domain:{domain}"
            suffix_hit = self.suffix_trie.match(domain)
            if suffix_hit:
                return f"suffix:{suffix_hit}"
        prefix_hit = self.prefix_matcher.match(text)
        if prefix_hit:
            return f"prefix:{prefix_hit}"
        glob_hit = self.glob_matcher.match(text)
        if glob_hit:
            return f"glob:{glob_hit}"
        return None
