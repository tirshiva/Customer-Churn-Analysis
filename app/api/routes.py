"""API routes."""

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from typing import Optional
import logging

from app.core.config import settings
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize templates
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Initialize prediction service
prediction_service = PredictionService()


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint - displays the prediction form.

    Args:
        request: FastAPI request object

    Returns:
        HTML response with prediction form
    """
    try:
        if not prediction_service.is_ready():
            logger.warning("Model not loaded, displaying error message")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "Model is not loaded. Please ensure the model file exists.",
                },
            )

        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering root page: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    """
    GET prediction endpoint - displays the form.

    Args:
        request: FastAPI request object

    Returns:
        HTML response with prediction form
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error in predict GET endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    tenure: Optional[int] = Form(None),
    monthly_charges: Optional[float] = Form(None),
    total_charges: Optional[float] = Form(None),
    gender: Optional[str] = Form(None),
    senior_citizen: Optional[str] = Form(None),
    partner: Optional[str] = Form(None),
    dependents: Optional[str] = Form(None),
    phone_service: Optional[str] = Form(None),
    multiple_lines: Optional[str] = Form(None),
    internet_service: Optional[str] = Form(None),
    online_security: Optional[str] = Form(None),
    online_backup: Optional[str] = Form(None),
    device_protection: Optional[str] = Form(None),
    tech_support: Optional[str] = Form(None),
    streaming_tv: Optional[str] = Form(None),
    streaming_movies: Optional[str] = Form(None),
    contract: Optional[str] = Form(None),
    paperless_billing: Optional[str] = Form(None),
    payment_method: Optional[str] = Form(None),
):
    """
    POST prediction endpoint - handles form submission and displays results.

    Args:
        request: FastAPI request object
        All form fields: Customer features from the form

    Returns:
        HTML response with prediction results or form with errors
    """
    try:
        # Validate that model is ready
        if not prediction_service.is_ready():
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "Model is not loaded. Please ensure the model file exists.",
                },
            )

        # Validate required fields
        if tenure is None or monthly_charges is None or total_charges is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error_message": (
                        "Please fill in all required fields "
                        "(Tenure, Monthly Charges, Total Charges)."
                    ),
                },
            )

        # Prepare input data
        input_data = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "gender": gender or "Male",
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner or "No",
            "Dependents": dependents or "No",
            "PhoneService": phone_service or "No",
            "MultipleLines": multiple_lines or "No",
            "InternetService": internet_service or "No",
            "OnlineSecurity": online_security or "No",
            "OnlineBackup": online_backup or "No",
            "DeviceProtection": device_protection or "No",
            "TechSupport": tech_support or "No",
            "StreamingTV": streaming_tv or "No",
            "StreamingMovies": streaming_movies or "No",
            "Contract": contract or "Month-to-month",
            "PaperlessBilling": paperless_billing or "No",
            "PaymentMethod": payment_method or "Electronic check",
        }

        # Make prediction
        prediction_result = prediction_service.predict(input_data)

        if prediction_result is None or prediction_result.get("status") == "error":
            error_msg = (
                prediction_result.get("message", "An error occurred during prediction")
                if prediction_result
                else "Prediction failed"
            )
            return templates.TemplateResponse(
                "error.html", {"request": request, "error_message": error_msg}
            )

        # Get feature importance
        feature_importance = prediction_service.get_feature_importance(top_n=10)

        # Prepare context for template
        context = {
            "request": request,
            "prediction": prediction_result,
            "input_data": input_data,
            "feature_importance": feature_importance,
        }

        return templates.TemplateResponse("result.html", context)

    except Exception as e:
        logger.error(f"Error in predict POST endpoint: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"An error occurred: {str(e)}"},
        )
