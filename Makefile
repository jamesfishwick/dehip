# Markdown here contains pinned tables and verbatim judge-prompt transcriptions.
# A formatter would reflow them. This no-op wins over mdformat in the global
# pre-commit hook, which prefers `make format` when the target exists.
.PHONY: format
format:
	@true
