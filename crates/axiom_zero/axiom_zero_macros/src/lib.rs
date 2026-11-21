use proc_macro::TokenStream;
use quote::quote;
use syn::visit::{self, Visit};
use syn::{
    parse_macro_input, Attribute, BinOp, Expr, ExprBinary, ExprCall, ExprIndex, ExprMethodCall,
    Item, ItemFn, ItemMod, Macro, Meta, PatIdent, Path,
};

struct AxiomZeroVisitor {
    violations: Vec<String>,
}

impl AxiomZeroVisitor {
    fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    fn has_test_attr(attrs: &[Attribute]) -> bool {
        for attr in attrs {
            if let Some(last) = attr.path().segments.last() {
                let ident = last.ident.to_string();
                if ident == "test" {
                    return true;
                }
                if ident == "cfg" {
                    if let Meta::List(list) = &attr.meta {
                        let has_test_tok = list
                            .tokens
                            .to_string()
                            .split(|c: char| !c.is_alphanumeric() && c != '_')
                            .any(|tok| tok == "test");
                        if has_test_tok {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn check_terminating_call(&mut self, path: &Path) {
        let segs: Vec<String> = path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect();

        let Some(last) = segs.last() else {
            return;
        };

        let is_bare_terminating =
            segs.len() == 1 && matches!(last.as_str(), "abort" | "exit" | "panic_any" | "_exit");
        let is_qualified_terminating = segs.len() >= 2
            && matches!(
                (segs[segs.len() - 2].as_str(), last.as_str()),
                ("process", "exit")
                    | ("process", "abort")
                    | ("intrinsics", "abort")
                    | ("panic", "panic_any")
                    | ("panicking", "panic")
                    | ("panicking", "begin_panic")
                    | ("libc", "abort")
                    | ("libc", "exit")
                    | ("libc", "_exit")
            );

        if is_bare_terminating || is_qualified_terminating {
            self.violations.push(
                "[E009] AxiomZero Violation: Process termination or raw panic call \
                 (`abort`/`exit`/`panic_any`) violates non-annihilating totality. \
                 Return a boundary `Presence<T>` or transition via `Attractor::absorb_perturbation`."
                    .to_string(),
            );
        }
    }
}

impl<'ast> Visit<'ast> for AxiomZeroVisitor {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        if Self::has_test_attr(&node.attrs) {
            return;
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if Self::has_test_attr(&node.attrs) {
            return;
        }
        visit::visit_item_fn(self, node);
    }

    // 1. Detect destructive or assertion macro invocations
    fn visit_macro(&mut self, node: &'ast Macro) {
        if let Some(last_seg) = node.path.segments.last() {
            let name = last_seg.ident.to_string();
            match name.as_str() {
                "panic" => {
                    self.violations.push(
                        "[E001] AxiomZero Violation: `panic!` invokes state annihilation. \
                         Replace with orbital relaxation (`Attractor::absorb_perturbation`) \
                         or return a boundary-guarded `Presence<T>`."
                            .to_string(),
                    );
                }
                "todo" => {
                    self.violations.push(
                        "[E002] AxiomZero Violation: `todo!` introduces an incomplete execution \
                         path. Complete all relational invariants prior to compilation."
                            .to_string(),
                    );
                }
                "unimplemented" => {
                    self.violations.push(
                        "[E003] AxiomZero Violation: `unimplemented!` introduces an incomplete \
                         execution path capable of runtime collapse."
                            .to_string(),
                    );
                }
                "unreachable" => {
                    self.violations.push(
                        "[E004] AxiomZero Violation: `unreachable!` crashes if control flow \
                         encounters an unexpected state. Map edge conditions into an explicit boundary."
                            .to_string(),
                    );
                }
                "assert"
                | "assert_eq"
                | "assert_ne"
                | "debug_assert"
                | "debug_assert_eq"
                | "debug_assert_ne" => {
                    self.violations.push(
                        "[E008] AxiomZero Violation: `assert!` / `debug_assert!` family invokes \
                         conditional panic. Use `RelationalIdentity::measure` or return a boundary `Presence<T>`."
                            .to_string(),
                    );
                }
                _ => {}
            }
        }
        visit::visit_macro(self, node);
    }

    // 2. Detect unsafe unwrapping methods
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method_name = node.method.to_string();
        match method_name.as_str() {
            "unwrap" | "unwrap_err" | "unwrap_unchecked" => {
                self.violations.push(
                    "[E005] AxiomZero Violation: `.unwrap()` / `.unwrap_err()` / `.unwrap_unchecked()` \
                     permits thread collapse or UB on None/Err. Use `Presence<T>` or `.unwrap_or(...)` / `.resolve(...)`."
                        .to_string(),
                );
            }
            "expect" | "expect_err" => {
                self.violations.push(
                    "[E006] AxiomZero Violation: `.expect()` / `.expect_err()` permits thread \
                     collapse on None/Err. Replace with `Presence::boundary(default)` or `.unwrap_or(...)`."
                        .to_string(),
                );
            }
            _ => {}
        }
        visit::visit_expr_method_call(self, node);
    }

    // 3. Detect terminating calls (`std::process::exit`, `abort`, etc.)
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(expr_path) = &*node.func {
            self.check_terminating_call(&expr_path.path);
        }
        visit::visit_expr_call(self, node);
    }

    // 4. Detect unchecked slice/array indexing `container[idx]`
    fn visit_expr_index(&mut self, node: &'ast ExprIndex) {
        self.violations.push(
            "[E010] AxiomZero Violation: Raw indexing `container[index]` can panic on out-of-bounds access. \
             Use `total_get`, `mask_get`, `TotalSliceExt::total_get`, or `Attractor::get_phase_state`."
                .to_string(),
        );
        visit::visit_expr_index(self, node);
    }

    // 5. Detect unchecked division `/` and modulo `%`
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        if matches!(
            node.op,
            BinOp::Div(_) | BinOp::DivAssign(_) | BinOp::Rem(_) | BinOp::RemAssign(_)
        ) {
            self.violations.push(
                "[E011] AxiomZero Violation: Unchecked division (`/`, `/=`) or modulo (`%`, `%=`) \
                 can trap on zero or produce NaN/Inf. Use `total_div_usize`, `total_rem_nonzero`, \
                 `total_div_f32`, or `total_div_unit_shifted_f32`."
                    .to_string(),
            );
        }
        visit::visit_expr_binary(self, node);
    }

    // 6. Detect direct references to `None` (`Option::None`, etc.) in paths and patterns
    fn visit_path(&mut self, node: &'ast Path) {
        if let Some(last_seg) = node.segments.last() {
            if last_seg.ident == "None" {
                self.violations.push(
                    "[E007] AxiomZero Violation: Reference to `None` (`Option::None`) attempts \
                     to instantiate void (Ø). Use `Presence::boundary(default)` or concrete fallback tokens."
                        .to_string(),
                );
            }
        }
        visit::visit_path(self, node);
    }

    fn visit_pat_ident(&mut self, node: &'ast PatIdent) {
        if node.ident == "None" {
            self.violations.push(
                "[E007] AxiomZero Violation: Pattern matching `None` references void (Ø). \
                 Use `Presence<T>` or wildcard fallback `_ => ...`."
                    .to_string(),
            );
        }
        visit::visit_pat_ident(self, node);
    }
}

/// Compile-time enforcement attribute implementing Axiom-Zero Constructive Totality rules:
/// `[E001]`–`[E011]`.
///
/// Rejects panics, assertions, `todo!`/`unimplemented!`/`unreachable!`, `.unwrap*()`/`.expect*()`,
/// direct `None` (`Option::None`) references, terminating calls (`exit`/`abort`),
/// unchecked indexing (`arr[i]`), and unchecked division/modulo (`/` and `%`).
/// Test functions (`#[test]`) and test modules (`#[cfg(test)]`) are exempted.
#[proc_macro_attribute]
pub fn axiom_zero(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let parsed_item = parse_macro_input!(item as Item);

    let mut visitor = AxiomZeroVisitor::new();
    visit::visit_item(&mut visitor, &parsed_item);

    if !visitor.violations.is_empty() {
        let error_msgs = visitor.violations.join("\n\n");
        let compile_error = quote! {
            compile_error!(#error_msgs);
        };
        return TokenStream::from(compile_error);
    }

    TokenStream::from(quote! {
        #parsed_item
    })
}
