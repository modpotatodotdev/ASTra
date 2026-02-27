use super::node_text;
use super::types::{ImportInfo, Symbol, SymbolKind};

pub(super) fn extract_rust_symbols(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<ImportInfo>,
    parent_scope: Option<&str>,
) {
    let kind = node.kind();
    match kind {
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                symbols.push(Symbol::from_node(
                    node,
                    name_node,
                    source,
                    file_path,
                    SymbolKind::Function,
                    parent_scope,
                ));
            }
        }
        "impl_item" => {
            // Extract the type name for the impl block
            let impl_name = node
                .child_by_field_name("type")
                .map(|n| node_text(n, source))
                .unwrap_or_default();

            // Recurse into the impl body for methods
            if let Some(body_node) = node.child_by_field_name("body") {
                let mut cursor = body_node.walk();
                for child in body_node.children(&mut cursor) {
                    if child.kind() == "function_item" {
                        extract_rust_symbols(
                            child,
                            source,
                            file_path,
                            symbols,
                            imports,
                            Some(&impl_name),
                        );
                    }
                }
            }
        }
        "use_declaration" => {
            let text = node_text(node, source);
            // Simple extraction of use path
            let module_path = text
                .trim_start_matches("use ")
                .trim_end_matches(';')
                .to_string();
            imports.push(ImportInfo::new(module_path, file_path));
        }
        _ => {}
    }
}
