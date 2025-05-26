from pydantic import BaseModel


class TextDataSchema(BaseModel):
    designation: str
    description: str
    productid: int
    imageid: int
    prdtypecode: int
