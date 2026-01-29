from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List

class TutorChatReq(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    user_input: str
    problem_id: str
    current_stage: str = "CONTEXT"
    history: List[List[str]] = []

class TutorChatRes(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

    ai_response: str
    current_stage: str
    intent: str
    is_validated: bool
    history: List[List[str]]

# class AnswerGuide(BaseModel):
#     model_config = ConfigDict(alias_generator=to_camel)
#
#     paragraph_type : str
#     original_text : str
    
class UserMsgCreateReq(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    
    user_id : int
    problem_id : int
    run_id : int
    user_message : str
    # session_id : int
    current_node : str
    
class UserMsgCreateRes(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )
    
    run_id : int
    status : str