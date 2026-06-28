//! The `#[reactive_scene]` attribute macro for the reactive BSN prototype
//! (`examples/scene/reactive_counter.rs`).
//!
//! A reactive scene's render function is a genuine Bevy system whose parameters are real
//! `SystemParam`s and which returns a `bsn!` scene. This macro turns it into a zero-arg
//! constructor and wires a reactive *dependency* per data-reading parameter:
//!
//! - `Res<T>`     -> `res_dep::<T>()`    (re-render when the resource changes)
//! - `Query<&C>`  -> `query_dep::<C>()`  (re-render when any `C` changes)
//!
//! ```ignore
//! #[reactive_scene]
//! fn panel(mut hooks: ReactiveHooks, score: Res<Score>, healths: Query<&Health>) -> impl Scene { .. }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, GenericArgument, ItemFn, PathArguments, Type};

#[proc_macro_attribute]
pub fn reactive_scene(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let vis = func.vis;
    let name = func.sig.ident;
    let inputs = func.sig.inputs;
    let body = *func.block;
    let hidden = format_ident!("__reactive_render_{}", name);

    // One reactive dependency per data-reading parameter.
    let deps: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pt) => dep_for(&pt.ty),
            _ => None,
        })
        .collect();

    quote! {
        fn #hidden(#inputs) -> ::std::boxed::Box<dyn Scene> {
            ::std::boxed::Box::new(#body)
        }

        #vis fn #name() -> ReactiveScene {
            reactive_scene_system(#hidden, ::std::vec![ #( #deps ),* ])
        }
    }
    .into()
}

/// Maps `Res<T>` -> `res_dep::<T>()` and `Query<&C>` -> `query_dep::<C>()`.
fn dep_for(ty: &Type) -> Option<proc_macro2::TokenStream> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|a| match a {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    });
    match segment.ident.to_string().as_str() {
        "Res" => {
            let inner = types.last()?;
            Some(quote! { res_dep::<#inner>() })
        }
        "Query" => match types.next()? {
            Type::Reference(reference) => {
                let component = &*reference.elem;
                Some(quote! { query_dep::<#component>() })
            }
            _ => None,
        },
        _ => None,
    }
}
