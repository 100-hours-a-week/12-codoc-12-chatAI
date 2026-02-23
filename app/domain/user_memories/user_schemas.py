from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List, Optional



class UserMemoryUpsertReq(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    user_id: int
    problem_id: Optional[int] = None
    created_at: int  # 이벤트 시각(에폭 초)
    text: str
    vector: List[float]
    tags: Optional[List[str]] = None
    seq: Optional[int] = None


class UserMemoryUpsertRes(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    point_id: str


class UserMemoryItem(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    user_id: int
    problem_id: Optional[int] = None
    created_at: int
    text: str
    tags: Optional[List[str]] = None
    seq: Optional[int] = None


class UserMemoryListRes(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    items: List[UserMemoryItem]
