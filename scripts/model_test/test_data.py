TEST_SCENARIOS = [
    # 1. BACKGROUND (배경 파악)
    {
        "step": "BACKGROUND", "level": "중급자", "is_correct": "True",
        "content": "메이플스토리라는 RPG 게임에는 레벨이 존재한다. 각각 만렙 = 300, 구만렙 = 275, 껌만렙 = 250이라는 용어가 존재한다.",
        "guide": "단순한 게임 스토리가 아니라, 데이터가 '특정 기준값(Threshold)'을 가진 정수들의 집합임을 인지시켜 주세요.",
        "keywords": "임계값(Threshold), 분류(Classification)",
        "user_msg": "데이터를 300, 275, 250이라는 임계값에 따라 네 가지 종류로 분류해야 합니다." # 스크린샷 Pupil 정답 적용
    },
    {
        "step": "BACKGROUND", "level": "중급자", "is_correct": "False",
        "content": "메이플스토리라는 RPG 게임에는 레벨이 존재한다. 각각 만렙 = 300, 구만렙 = 275, 껌만렙 = 250이라는 용어가 존재한다.",
        "guide": "단순한 게임 스토리가 아니라, 데이터가 '특정 기준값(Threshold)'을 가진 정수들의 집합임을 인지시켜 주세요.",
        "keywords": "임계값(Threshold), 분류(Classification)",
        "user_msg": "입력받은 레벨 데이터들을 리스트에 담아서 정렬하는 것이 핵심입니다." # 스크린샷 Pupil 오답 적용
    },

    # 2. GOAL (목표 설정)
    {
        "step": "GOAL", "level": "중급자", "is_correct": "True",
        "content": "N개의 레벨 M이 주어질 때, 각 레벨이 속한 구간의 번호를 출력해 보자.",
        "guide": "단순 출력이 아니라, 입력된 수치(Level)를 정해진 카테고리(1~4)로 변환(Mapping)하는 것이 최종 목표임을 알려주세요.",
        "keywords": "구간 번호(Section ID), 매핑(Mapping)",
        "user_msg": "사용자 레벨을 확인해서 그에 맞는 구간 번호를 결과로 매핑해서 내보내는 것입니다." # 스크린샷 Pupil 정답 적용
    },
    {
        "step": "GOAL", "level": "중급자", "is_correct": "False",
        "content": "N개의 레벨 M이 주어질 때, 각 레벨이 속한 구간의 번호를 출력해 보자.",
        "guide": "단순 출력이 아니라, 입력된 수치(Level)를 정해진 카테고리(1~4)로 변환(Mapping)하는 것이 최종 목표임을 알려주세요.",
        "keywords": "구간 번호(Section ID), 매핑(Mapping)",
        "user_msg": "사용자가 입력한 레벨 값들을 그대로 다시 출력해주는 것이 목표입니다." # 스크린샷 Pupil 오답 적용
    },

    # 3. STRATEGY (전략 수립)
    {
        "step": "STRATEGY", "level": "중급자", "is_correct": "True",
        "content": "모든 레벨은 만렙(구간 1), 구만렙이상 만렙 미만(구간 2), 껌만렙이상 구만렙 미만(구간 3), 껌만렙 미만(구간 4)에 속하게 된다.",
        "guide": "구간이 명확히 나뉘어 있으므로, 특정 범위를 체크하는 '다중 조건문(if-else if)' 기법을 사용해야 함을 힌트로 주세요.",
        "keywords": "조건문 분기(Conditional Branching), if-else if",
        "user_msg": "if-else if 문을 사용하여 위에서부터 차례대로 조건문 분기 처리를 하겠습니다." # 스크린샷 Pupil 정답 적용
    },
    {
        "step": "STRATEGY", "level": "중급자", "is_correct": "False",
        "content": "모든 레벨은 만렙(구간 1), 구만렙이상 만렙 미만(구간 2), 껌만렙이상 구만렙 미만(구간 3), 껌만렙 미만(구간 4)에 속하게 된다.",
        "guide": "구간이 명확히 나뉘어 있으므로, 특정 범위를 체크하는 '다중 조건문(if-else if)' 기법을 사용해야 함을 힌트로 주세요.",
        "keywords": "조건문 분기(Conditional Branching), if-else if",
        "user_msg": "for 문을 써서 모든 숫자를 다 더한 다음에 평균을 내서 비교하겠습니다." # 스크린샷 Pupil 오답 적용
    },

    # 4. INSIGHT (예외 및 통찰)
    {
        "step": "INSIGHT", "level": "중급자", "is_correct": "True",
        "content": "첫 번째 줄에는 레벨의 개수 N(1 ≤ N ≤ 100)이 주어진다. 두 번째 줄에는 N개의 레벨 M(1 ≤ M ≤ 300)이 공백으로 구분되어 주어진다.",
        "guide": "N이 최대 100으로 매우 작으므로, 복잡한 최적화 없이 단순 반복문 O(N)으로도 시간 내에 충분히 해결 가능함을 검증해주세요.",
        "keywords": "선형 탐색(Linear Search), O(N)",
        "user_msg": "입력 개수 N이 최대 100개라 O(N)으로도 충분히 돌아가니 선형 탐색으로 풀겠습니다." # 스크린샷 Pupil 정답 적용
    },
    {
        "step": "INSIGHT", "level": "중급자", "is_correct": "False",
        "content": "첫 번째 줄에는 레벨의 개수 N(1 ≤ N ≤ 100)이 주어진다. 두 번째 줄에는 N개의 레벨 M(1 ≤ M ≤ 300)이 공백으로 구분되어 주어진다.",
        "guide": "N이 최대 100으로 매우 작으므로, 복잡한 최적화 없이 단순 반복문 O(N)으로도 시간 내에 충분히 해결 가능함을 검증해주세요.",
        "keywords": "선형 탐색(Linear Search), O(N)",
        "user_msg": "정렬을 한 다음에 이분 탐색을 써야만 시간 초과가 안 날 것 같아요." # 스크린샷 Pupil 오답 적용
    }
]