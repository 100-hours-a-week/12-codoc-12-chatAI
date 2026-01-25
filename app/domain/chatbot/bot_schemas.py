from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List

class AnswerGuide(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    
    paragraph_type : str
    original_text : str
    
class UserMsgCreateReq(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    
    user_id : int
    problem_id : int
    run_id : int
    user_message : str
    # session_id : int
    current_node : str
    answer_guides: List[AnswerGuide]
    
class UserMsgCreateRes(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )
    
    run_id : int
    status : str