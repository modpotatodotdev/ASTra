mod js_parser;
mod python_parser;
mod rust_parser;
mod types;

pub use types::{ImportInfo, Language, ParsedFile, Symbol, SymbolKind};

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};

use js_parser::extract_js_symbols;
use python_parser::extract_python_symbols;
use rust_parser::extract_rust_symbols;

/// Parse a source file and extract symbols using tree-sitter.
pub fn parse_file(file_path: &Path, workspace_root: &Path) -> Result<ParsedFile> {
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;

    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let lang = Language::from_extension(ext)
        .with_context(|| format!("unsupported file extension: {}", ext))?;

    let relative_path = file_path
        .strip_prefix(workspace_root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string();

    parse_source(&source, lang, &relative_path)
}

/// Parse source code string and extract symbols.
pub fn parse_source(source: &str, lang: Language, file_path: &str) -> Result<ParsedFile> {
    let ts_language = match lang {
        Language::Rust => tree_sitter_rust::language(),
        Language::Python => tree_sitter_python::language(),
        Language::JavaScript => tree_sitter_javascript::language(),
        Language::TypeScript => tree_sitter_typescript::language_typescript(),
    };

    let mut ts_parser = tree_sitter::Parser::new();
    ts_parser
        .set_language(&ts_language)
        .context("failed to set tree-sitter language")?;

    let tree = ts_parser
        .parse(source, None)
        .context("failed to parse source")?;

    let root = tree.root_node();
    let mut symbols = Vec::new();
    let mut imports = Vec::new();

    extract_symbols(
        root,
        source,
        file_path,
        lang,
        &mut symbols,
        &mut imports,
        None,
    );

    static CALL_PATTERN: OnceLock<regex::Regex> = OnceLock::new();
    let call_pattern =
        CALL_PATTERN.get_or_init(|| regex::Regex::new(r"(?P<ident>[a-zA-Z_]\w*)\s*\(").unwrap());
    for symbol in &mut symbols {
        let mut calls = Vec::new();
        for cap in call_pattern.captures_iter(&symbol.body) {
            if let Some(ident) = cap.name("ident") {
                let name = ident.as_str();
                if name != symbol.name {
                    calls.push(name.to_string());
                }
            }
        }
        calls.sort();
        calls.dedup();
        symbol.calls = calls;
    }

    Ok(ParsedFile {
        path: file_path.to_string(),
        symbols,
        imports,
    })
}

/// Recursively extract symbols from the AST.
fn extract_symbols(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    lang: Language,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<ImportInfo>,
    parent_scope: Option<&str>,
) {
    let kind = node.kind();

    match lang {
        Language::Rust => {
            extract_rust_symbols(node, source, file_path, symbols, imports, parent_scope)
        }
        Language::Python => {
            extract_python_symbols(node, source, file_path, symbols, imports, parent_scope)
        }
        Language::JavaScript | Language::TypeScript => {
            extract_js_symbols(node, source, file_path, symbols, imports, parent_scope)
        }
    }

    // Recurse into children for nodes that aren't themselves symbols
    if !is_symbol_node(kind, lang) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            extract_symbols(
                child,
                source,
                file_path,
                lang,
                symbols,
                imports,
                parent_scope,
            );
        }
    }
}

fn is_symbol_node(kind: &str, lang: Language) -> bool {
    match lang {
        Language::Rust => matches!(
            kind,
            "function_item" | "impl_item" | "struct_item" | "use_declaration"
        ),
        Language::Python => matches!(
            kind,
            "function_definition"
                | "class_definition"
                | "import_statement"
                | "import_from_statement"
        ),
        Language::JavaScript | Language::TypeScript => matches!(
            kind,
            "function_declaration"
                | "arrow_function"
                | "method_definition"
                | "class_declaration"
                | "import_statement"
                | "export_statement"
        ),
    }
}

/// Extract the text content of an AST node from source.
fn node_text(node: tree_sitter::Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

#[cfg(test)]
mod tests {
    use super::{parse_source, Language, SymbolKind};

    #[test]
    fn test_parse_rust_function() {
        let source = r#"
fn hello() {
    println!("hello world");
}

fn greet(name: &str) {
    hello();
    println!("hi {}", name);
}
"#;
        let result = parse_source(source, Language::Rust, "test.rs").unwrap();
        assert_eq!(result.symbols.len(), 2);
        assert_eq!(result.symbols[0].name, "hello");
        assert_eq!(result.symbols[0].kind, SymbolKind::Function);
        assert_eq!(result.symbols[1].name, "greet");
    }

    #[test]
    fn test_parse_python_class() {
        let source = r#"
class UserService:
    def create_user(self, name):
        return self.save(name)

    def save(self, name):
        pass
"#;
        let result = parse_source(source, Language::Python, "service.py").unwrap();
        // Should find the class and two methods
        let class_symbols: Vec<_> = result
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Class)
            .collect();
        let method_symbols: Vec<_> = result
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Method)
            .collect();
        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].name, "UserService");
        assert_eq!(method_symbols.len(), 2);
    }

    #[test]
    fn test_parse_javascript_function() {
        let source = r#"
function fetchData(url) {
    return fetch(url);
}

function processData() {
    const data = fetchData("/api");
    return data;
}
"#;
        let result = parse_source(source, Language::JavaScript, "app.js").unwrap();
        assert_eq!(result.symbols.len(), 2);
        assert_eq!(result.symbols[0].name, "fetchData");
        assert_eq!(result.symbols[1].name, "processData");
    }

    #[test]
    fn test_parse_javascript_arrow_function_variable() {
        let source = r#"
const handler = (req, res) => {
    return processReq(req);
}
"#;
        let result = parse_source(source, Language::JavaScript, "app.js").unwrap();
        assert_eq!(result.symbols.len(), 1);
        assert_eq!(result.symbols[0].name, "handler");
        assert_eq!(result.symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn test_parse_exported_javascript_arrow_function_variable() {
        let source = r#"
export const handler = (req, res) => {
    return processReq(req);
}
"#;
        let result = parse_source(source, Language::JavaScript, "app.js").unwrap();
        assert_eq!(result.symbols.len(), 1);
        assert_eq!(result.symbols[0].name, "handler");
        assert_eq!(result.symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("js"), Some(Language::JavaScript));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("txt"), None);
    }

    #[test]
    fn test_scoped_symbol_ids_avoid_method_name_collision() {
        let source = r#"
struct Foo;
impl Foo { fn new() -> Self { Foo } }
struct Bar;
impl Bar { fn new() -> Self { Bar } }
"#;
        let result = parse_source(source, Language::Rust, "lib.rs").unwrap();
        let ids: Vec<&str> = result
            .symbols
            .iter()
            .filter(|s| s.name == "new")
            .map(|s| s.id.as_str())
            .collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().any(|id| *id == "lib.rs::Foo::new"));
        assert!(ids.iter().any(|id| *id == "lib.rs::Bar::new"));
    }
}
