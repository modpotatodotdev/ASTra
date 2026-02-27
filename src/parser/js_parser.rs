use super::node_text;
use super::types::{ImportInfo, Symbol, SymbolKind};

pub(super) fn extract_js_symbols(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<ImportInfo>,
    parent_scope: Option<&str>,
) {
    let kind = node.kind();
    match kind {
        "function_declaration" => {
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
        "class_declaration" => {
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
                        if child.kind() == "method_definition" {
                            extract_js_symbols(
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
        }
        "method_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                symbols.push(Symbol::from_node(
                    node,
                    name_node,
                    source,
                    file_path,
                    SymbolKind::Method,
                    parent_scope,
                ));
            }
        }
        "variable_declarator" => {
            if let (Some(name_node), Some(value_node)) = (
                node.child_by_field_name("name"),
                node.child_by_field_name("value"),
            ) {
                if matches!(value_node.kind(), "arrow_function" | "function") {
                    symbols.push(Symbol::from_node(
                        value_node,
                        name_node,
                        source,
                        file_path,
                        SymbolKind::Function,
                        parent_scope,
                    ));
                }
            }
        }
        "import_statement" => {
            let text = node_text(node, source);
            imports.push(ImportInfo::new(text, file_path));
        }
        "export_statement" => {
            // Check if there's a declaration inside the export
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "function_declaration" | "class_declaration" | "variable_declarator" => {
                        extract_js_symbols(
                            child,
                            source,
                            file_path,
                            symbols,
                            imports,
                            parent_scope,
                        );
                    }
                    "lexical_declaration" | "variable_declaration" => {
                        let mut decl_cursor = child.walk();
                        for decl_child in child.children(&mut decl_cursor) {
                            extract_js_symbols(
                                decl_child,
                                source,
                                file_path,
                                symbols,
                                imports,
                                parent_scope,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}
