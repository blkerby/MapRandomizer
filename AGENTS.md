General guidelines:
- Ask for confirmation before diving into making changes, unless the exact
  desired code change is already clear from the user's request. This includes
  follow-up debugging after a reported regression: investigate and explain the
  likely cause first, but do not apply a speculative fix or design change until
  the user confirms that specific change.
- Unless the user specifies otherwise, any tests added to verify changes should
  be treated as temporary and removed before completing the task.

Rust style:
- Do not convert `bool` to integer types using casts or `from` conversions.
  Instead use if-else expressions.