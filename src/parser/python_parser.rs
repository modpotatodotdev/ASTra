use super::node_text;
use super::types::{ImportInfo, Symbol, SymbolKind};

pub(super) fn extract_python_symbols(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<ImportInfo>,
    parent_scope: Option<&str>,
) {
    let kind = node.kind();
    match kind {
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let sym_kind = if parent_scope.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                symbols.push(Symbol::from_node(
                    node,
                    name_node,
                    source,
                    file_path,
                    sym_kind,
                    parent_scope,
                ));
            }
        }
        "class_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let sym = Symbol::from_node(
                    node,
                    name_node,
                    source,
                    file_path,
                    SymbolKind::Class,
                    parent_scope,
                );
                let class_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into class body for methods
                if let Some(body_node) = node.child_by_field_name("body") {
                    let mut cursor = body_node.walk();
                    for child in body_node.children(&mut cursor) {
                        extract_python_symbols(
                            child,
                            source,
                            file_path,
                            symbols,
                            imports,
                            Some(&class_name),
                        );
                    }
                }
            }
        }
        "import_statement" | "import_from_statement" => {
            let text = node_text(node, source);
            imports.push(ImportInfo::new(text, file_path));
        }
        _ => {}
    }
}
