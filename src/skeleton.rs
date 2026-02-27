/// Extract the signature and leading doc-comment from a symbol body.
///
/// Returns up to 6 lines of the opening signature plus any immediately
/// following doc-comment block (Python triple-quoted strings, `///`-style
/// comments, `/* */` blocks, `#` comments, etc.), capped at 14 doc lines.
pub fn build_skeleton_context(body: &str) -> String {
    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut start = 0;
    while start < lines.len() && lines[start].trim().is_empty() {
        start += 1;
    }
    if start >= lines.len() {
        return String::new();
    }

    let mut signature_end = start;
    let mut captured = 0;
    while signature_end < lines.len() && captured < 6 {
        let line = lines[signature_end];
        captured += 1;
        signature_end += 1;
        let trimmed = line.trim_end();
        if trimmed.contains('{') || trimmed.ends_with(':') {
            break;
        }
    }

    let mut out: Vec<&str> = lines[start..signature_end.min(lines.len())].to_vec();

    let mut idx = signature_end;
    while idx < lines.len() && lines[idx].trim().is_empty() {
        idx += 1;
    }

    if idx >= lines.len() {
        return out.join("\n");
    }

    let first = lines[idx].trim_start();
    let mut doc_lines: Vec<&str> = Vec::new();

    if first.starts_with("\"\"\"") || first.starts_with("'''") {
        let delimiter = if first.starts_with("\"\"\"") {
            "\"\"\""
        } else {
            "'''"
        };
        let mut first_seen = false;
        while idx < lines.len() && doc_lines.len() < 14 {
            let line = lines[idx];
            doc_lines.push(line);
            let trimmed = line.trim_start();
            let occurrences = trimmed.matches(delimiter).count();
            if occurrences > 0 {
                if first_seen {
                    break;
                }
                if occurrences >= 2 {
                    break;
                }
                first_seen = true;
            }
            idx += 1;
        }
    } else if is_doc_comment_line(first) {
        let mut in_block_comment = first.starts_with("/*") && !first.contains("*/");
        while idx < lines.len() && doc_lines.len() < 14 {
            let line = lines[idx];
            let trimmed = line.trim_start();
            if !in_block_comment && !is_doc_comment_line(trimmed) {
                break;
            }
            doc_lines.push(line);
            if in_block_comment && trimmed.contains("*/") {
                in_block_comment = false;
            } else if trimmed.starts_with("/*") && !trimmed.contains("*/") {
                in_block_comment = true;
            }
            idx += 1;
        }
    }

    if !doc_lines.is_empty() {
        out.push("");
        out.extend(doc_lines);
    }

    out.join("\n")
}

fn is_doc_comment_line(line: &str) -> bool {
    line.starts_with("///")
        || line.starts_with("//!")
        || line.starts_with("//")
        || line.starts_with('#')
        || line.starts_with("/*")
        || line.starts_with('*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_body() {
        assert_eq!(build_skeleton_context(""), "");
    }

    #[test]
    fn test_whitespace_only_body() {
        assert_eq!(build_skeleton_context("   \n  \n  "), "");
    }

    #[test]
    fn test_rust_function_signature() {
        let body = "fn fetch_user(id: u64) -> User {\n    db.query(id)\n}";
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("fn fetch_user(id: u64) -> User {"));
        assert!(!skeleton.contains("db.query"));
    }

    #[test]
    fn test_rust_function_with_doc_comment() {
        let body = r#"/// Fetch a user by ID from the database.
///
/// Returns None if the user doesn't exist.
fn fetch_user(id: u64) -> Option<User> {
    db.query(id)
}"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("fn fetch_user(id: u64) -> Option<User> {"));
        assert!(skeleton.contains("/// Fetch a user by ID"));
        assert!(skeleton.contains("/// Returns None if"));
        assert!(!skeleton.contains("db.query"));
    }

    #[test]
    fn test_python_function_with_docstring() {
        let body = r#"def fetch_user(id: int) -> User:
    """Fetch a user by ID from the database.

    Returns None if the user doesn't exist.
    """
    return db.query(id)"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("def fetch_user(id: int) -> User:"));
        assert!(skeleton.contains("\"\"\"Fetch a user"));
        assert!(!skeleton.contains("return db.query"));
    }

    #[test]
    fn test_python_function_with_single_quotes_docstring() {
        let body = r#"def fetch_user(id):
    '''Fetch a user by ID.'''
    return db.query(id)"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("def fetch_user(id):"));
        assert!(skeleton.contains("'''Fetch a user by ID.'''"));
    }

    #[test]
    fn test_python_multiline_docstring_closes_same_line() {
        let body = r#"def helper():
    """Does something useful."""
    return 42"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("def helper():"));
        assert!(skeleton.contains("\"\"\"Does something useful.\"\"\""));
    }

    #[test]
    fn test_javascript_function_with_comment() {
        let body = "function calculateTotal(amount) {\n    return amount * 1.1;\n}";
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("function calculateTotal(amount) {"));
        assert!(!skeleton.contains("return amount"));
    }

    #[test]
    fn test_rust_impl_methods() {
        let body = r#"/// Get the user's display name.
pub fn name(&self) -> &str {
    &self.display_name
}"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("pub fn name(&self) -> &str {"));
        assert!(skeleton.contains("/// Get the user's display name."));
        assert!(!skeleton.contains("&self.display_name"));
    }

    #[test]
    fn test_signature_breaks_on_brace() {
        let body = "fn foo() {\n    body\n}";
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("fn foo() {"));
        assert!(!skeleton.contains("body"));
    }

    #[test]
    fn test_signature_breaks_on_colon() {
        let body = "def foo():\n    body";
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("def foo():"));
        assert!(!skeleton.contains("body"));
    }

    #[test]
    fn test_blank_lines_before_signature_skipped() {
        let body = "\n\n\nfn foo() {\n    body\n}";
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("fn foo() {"));
        assert!(!skeleton.starts_with('\n'));
    }

    #[test]
    fn test_no_doc_comment_after_signature() {
        let body = "fn foo() {\n    body\n}";
        let skeleton = build_skeleton_context(body);
        assert_eq!(skeleton, "fn foo() {");
    }

    #[test]
    fn test_hash_comments_captured() {
        let body = r#"# Initialize the database connection.
# Raises ConnectionError if unreachable.
def init_db():
    connect()"#;
        let skeleton = build_skeleton_context(body);
        assert!(skeleton.contains("# Initialize the database"));
        assert!(skeleton.contains("# Raises ConnectionError"));
        assert!(skeleton.contains("def init_db():"));
    }

    #[test]
    fn test_doc_comment_cap_at_14_lines() {
        let doc_lines: String = (0..20)
            .map(|i| format!("/// Doc line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let body = format!("fn foo() {{\n{}\n    body\n}}", doc_lines);
        let skeleton = build_skeleton_context(&body);
        let skeleton_doc_lines: usize = skeleton
            .lines()
            .filter(|l| l.trim_start().starts_with("///"))
            .count();
        assert!(
            skeleton_doc_lines <= 14,
            "expected at most 14 doc lines, got {}",
            skeleton_doc_lines
        );
    }
}
