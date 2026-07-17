from app.database import SessionLocal, create_tables
from app.models import ProfileModel, ProjectModel


PROFILE_ID = 1


PROFILE = {
    "name": "陈若嘉",
    "headline": "计算机本科生 / 后端与 AI 应用开发方向",
    "email": "ruojia.chen@example.com",
    "phone": "+61 412 345 678",
    "wechat": "",
    "location": "Sydney, Australia",
    "summary": (
        "计算机科学本科生，关注后端开发、数据分析和 AI 应用落地。具备 Java、Python、R、SQL "
        "和基础前端开发经验，完成过课程管理系统、数据分析项目、时间序列预测研究和职业规划 Agent 原型。"
        "希望申请 Junior Backend Developer、Graduate Software Engineer 或 AI Application Developer 相关岗位。"
    ),
    "skills": [
        "Java",
        "Spring Boot",
        "Python",
        "FastAPI",
        "R",
        "SQL",
        "PostgreSQL",
        "React",
        "TypeScript",
        "Pandas",
        "Prophet",
        "Time Series Forecasting",
        "LangChain",
        "ChromaDB",
        "REST API",
        "Git",
    ],
    "education": [],
    "experience": [],
}


PROJECTS = [
    {
        "id": "seed-campus-course-booking",
        "category": "project",
        "title": "校园课程预约管理系统",
        "role": "Java 后端开发",
        "start_date": "2025-03",
        "end_date": "2025-06",
        "description": (
            "设计并实现一个面向学生的课程预约管理系统，支持用户注册登录、课程浏览、预约、取消预约和管理员课程管理。"
            "我主要负责后端业务逻辑、接口设计、数据校验和异常处理。"
        ),
        "technologies": ["Java", "Spring Boot", "MySQL", "REST API", "JPA", "Git"],
        "highlights": [
            "实现用户、课程、预约记录等核心模块",
            "设计预约容量、时间冲突和用户权限校验逻辑",
            "使用 REST API 支撑前后端数据交互",
            "提升了对后端分层结构和业务规则建模的理解",
        ],
    },
    {
        "id": "seed-student-performance-analysis",
        "category": "project",
        "title": "学生成绩与学习行为数据分析",
        "role": "R 数据分析员",
        "start_date": "2024-09",
        "end_date": "2024-11",
        "description": (
            "基于学生成绩、出勤率、作业提交情况和课程参与数据，使用 R 进行清洗、统计分析和可视化，"
            "探索影响学生成绩表现的主要因素。我负责数据预处理、描述性统计、相关性分析和图表生成。"
        ),
        "technologies": ["R", "tidyverse", "ggplot2", "dplyr", "R Markdown"],
        "highlights": [
            "完成缺失值处理和异常值检查",
            "使用可视化展示出勤率、作业完成度和最终成绩之间的关系",
            "输出 R Markdown 分析报告",
            "提升了数据解释和报告表达能力",
        ],
    },
    {
        "id": "seed-housing-price-prediction",
        "category": "project",
        "title": "二手房价格预测分析",
        "role": "Python 数据开发",
        "start_date": "2024-07",
        "end_date": "2024-09",
        "description": (
            "使用 Python 对房屋交易数据进行清洗、特征处理和价格预测建模，分析地理位置、面积、房型和配套设施对价格的影响。"
            "我主要负责数据处理、特征工程、模型训练和结果评估。"
        ),
        "technologies": ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "SQL"],
        "highlights": [
            "构建数据清洗和特征处理流程",
            "使用线性回归和随机森林进行价格预测",
            "对模型误差进行对比分析",
            "通过图表解释主要影响因素",
        ],
    },
    {
        "id": "seed-time-series-model-comparison",
        "category": "project",
        "title": "时间序列预测模型对比研究",
        "role": "时间序列模型研究员",
        "start_date": "2025-02",
        "end_date": "2025-05",
        "description": (
            "对多个时间序列预测模型进行实验对比，研究 Prophet、ARIMA 和深度学习模型在不同数据集上的预测表现。"
            "我负责 Prophet 模型实验、参数调整、结果记录和误差分析。"
        ),
        "technologies": ["Python", "Prophet", "Pandas", "NumPy", "Matplotlib", "Time Series Forecasting"],
        "highlights": [
            "完成多组预测实验和结果对比",
            "使用 MAE、RMSE、MAPE 评估模型表现",
            "分析不同预测步长下模型稳定性",
            "整理实验结果用于项目报告和展示",
        ],
    },
    {
        "id": "seed-ai-career-agent",
        "category": "project",
        "title": "AI 职业规划 Agent",
        "role": "AI 应用开发",
        "start_date": "2025-08",
        "end_date": "2025-11",
        "description": (
            "设计一个面向求职学生的 AI 职业规划 Agent，根据用户信息、项目经历和岗位库，生成职业方向推荐、"
            "简历版本、岗位匹配结果、技能差距分析和学习计划。我负责 Agent 流程设计、工具拆分、岗位排序逻辑和结构化输出。"
        ),
        "technologies": ["Python", "FastAPI", "LangChain", "ChromaDB", "PostgreSQL", "OpenAI API", "Pydantic"],
        "highlights": [
            "将职业规划流程拆分为项目画像、方向推荐、简历生成、岗位排序和学习计划多个工具",
            "设计 Agent 状态管理和反馈重跑逻辑",
            "结合关键词匹配、向量召回和规则分数进行岗位排序",
            "输出可解释的 Top10 岗位推荐结果",
        ],
    },
]


def main() -> None:
    create_tables()
    with SessionLocal() as db:
        profile = db.get(ProfileModel, PROFILE_ID)
        if profile is None:
            profile = ProfileModel(id=PROFILE_ID)
            db.add(profile)

        for field, value in PROFILE.items():
            setattr(profile, field, value)

        for project_data in PROJECTS:
            project = db.get(ProjectModel, project_data["id"])
            if project is None:
                project = ProjectModel(id=project_data["id"], profile_id=PROFILE_ID)
                db.add(project)

            project.profile_id = PROFILE_ID
            for field, value in project_data.items():
                setattr(project, field, value)

        db.commit()

    print(f"Seeded profile and {len(PROJECTS)} projects.")


if __name__ == "__main__":
    main()
