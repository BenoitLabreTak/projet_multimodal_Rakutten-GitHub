from fastapi import APIRouter
from app.services.run_tests import run_pytest_test

router = APIRouter()

@router.get("/preprocess/text/manual")
def test_preproces_text():
    return run_pytest_test("tests/test_api_preprocess_text.py")

@router.get("/preprocess/image")
def test_preprocess_image():
    return run_pytest_test("tests/test_api_preprocess_image.py")

@router.get("/predict/text/manual")
def test_text_manual():
    return run_pytest_test("tests/test_api_predict_text_manual.py")

@router.get("/predict/text/file")
def test_text_file():
    return run_pytest_test("tests/test_api_predict_text_file.py")

@router.get("/predict/image/manual")
def test_image_single():
    return run_pytest_test("tests/test_api_predict_image_single.py")

@router.get("/predict/image/file")
def test_image_file():
    return run_pytest_test("tests/test_api_predict_image_file.py")