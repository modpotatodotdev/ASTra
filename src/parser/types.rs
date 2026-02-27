use serde::{Deserialize, Serialize};

/// A symbol extracted from a source file (function, method, class, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Symbol {
    /// Unique identifier: `file_path::name` (or `file_path::parent_scope::name` for scoped symbols)
    pub id: String,
    /// Human-readable name of the symbol.
    pub name: String,
    /// The kind of symbol.
    pub kind: SymbolKind,
    /// File path relative to the workspace root.
    pub file_path: String,
    /// Byte range in the source file.
    pub byte_range: (usize, usize),
    /// Line range (start_line, end_line) — 0-indexed.
    pub line_range: (usize, usize),
    /// The source code body of this symbol.
    pub body: String,
    /// Names of other symbols that this symbol calls/references.
    pub calls: Vec<String>,
    /// The parent scope (e.g., the class or module containing this function).
    pub parent_scope: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Module,
    Import,
}

/// Result of parsing a single file.
#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub path: String,
    pub symbols: Vec<Symbol>,
    pub imports: Vec<ImportInfo>,
}

/// Import information extracted from a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    pub module_path: String,
    pub imported_names: Vec<String>,
    pub file_path: String,
}

impl Symbol {
    /// Create a Symbol from a tree-sitter node.
    pub(crate) fn from_node(
        node: tree_sitter::Node,
        name_node: tree_sitter::Node,
        source: &str,
        file_path: &str,
        kind: SymbolKind,
        parent_scope: Option<&str>,
    ) -> Self {
        let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
        let body = source[node.start_byte()..node.end_byte()].to_string();
        let id = if let Some(scope) = parent_scope {
            format!("{}::{}::{}", file_path, scope, name)
        } else {
            format!("{}::{}", file_path, name)
        };
        Self {
            id,
            name,
            kind,
            file_path: file_path.to_string(),
            byte_range: (node.start_byte(), node.end_byte()),
            line_range: (node.start_position().row, node.end_position().row),
            body,
            calls: Vec::new(),
            parent_scope: parent_scope.map(|s| s.to_string()),
        }
    }
}

impl ImportInfo {
    pub(crate) fn new(module_path: String, file_path: &str) -> Self {
        Self {
            module_path,
            imported_names: Vec::new(),
            file_path: file_path.to_string(),
        }
    }
}

/// Supported languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
}

impl Language {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "js" | "jsx" => Some(Language::JavaScript),
            "ts" | "tsx" => Some(Language::TypeScript),
            _ => None,
        }
    }
}
