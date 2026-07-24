// Small wasm-only browser helpers shared by the screenshot and recorder
// paths, so neither feature depends on the other.

use wasm_bindgen::JsCast;

/// The app's render canvas (id set in web/index.html and the Window config).
pub fn canvas() -> Option<web_sys::HtmlCanvasElement> {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("fractality-canvas"))
        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
}

/// Trigger a browser download of a blob via a synthetic anchor click.
pub fn download_blob(blob: &web_sys::Blob, name: &str) {
    let Ok(url) = web_sys::Url::create_object_url_with_blob(blob) else {
        return;
    };
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Ok(link) = doc.create_element("a") {
            let _ = link.set_attribute("href", &url);
            let _ = link.set_attribute("download", name);
            if let Ok(el) = link.dyn_into::<web_sys::HtmlElement>() {
                el.click();
            }
        }
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}
