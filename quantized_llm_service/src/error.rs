use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("model is still loading")]
    ModelLoading,
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("model execution failed: {0}")]
    Inference(String),
    #[error("quantization error: {0}")]
    Quantization(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("other: {0}")]
    Other(String),
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status = match self {
            ServiceError::ModelLoading => StatusCode::SERVICE_UNAVAILABLE,
            ServiceError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ServiceError::Tokenizer(_)
            | ServiceError::Inference(_)
            | ServiceError::Quantization(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServiceError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServiceError::Other(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = serde_json::json!({
            "error": self.to_string(),
        });

        (status, axum::Json(body)).into_response()
    }
}
