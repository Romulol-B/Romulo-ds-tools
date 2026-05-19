from __future__ import annotations

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - compatibility for older Pandera releases.
    import pandera as pa


def get_schema() -> pa.DataFrameSchema:
    """Return the raw input data contract for student_performace."""
    return pa.DataFrameSchema(
        {
            "student_id": pa.Column(str, nullable=True, coerce=True),
            "timestamp": pa.Column(str, nullable=True, coerce=True),
            "year_class": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Final Year",
                        "First Year (FY)",
                        "First Year (PG)",
                        "Second Year (PG)",
                        "Second Year (SY)",
                        "Third Year (TY)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "program_stream": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "BA",
                        "BBA",
                        "BCA",
                        "BCom",
                        "BSc Computer Science",
                        "BSc Cyber Security",
                        "BSc IT",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "age": pa.Column(int, nullable=True, coerce=True),
            "gender": pa.Column(
                str, checks=pa.Check.isin(["Female", "Male", "Other"]), nullable=True, coerce=True
            ),
            "cgpa_category": pa.Column(
                str,
                checks=pa.Check.isin(["5.0 – 6.9", "7.0 – 8.4", "8.5 – 9.4", "9.5 – 10.0"]),
                nullable=True,
                coerce=True,
            ),
            "academic_satisfaction": pa.Column(
                str,
                checks=pa.Check.isin(
                    ["Neutral", "Satisfied", "Unsatisfied", "Very satisfied", "Very unsatisfied"]
                ),
                nullable=True,
                coerce=True,
            ),
            "study_hours_daily": pa.Column(
                str,
                checks=pa.Check.isin(["1–2 hours", "Less than 1 hour", "More than 2 hours"]),
                nullable=True,
                coerce=True,
            ),
            "daily_productivity": pa.Column(float, nullable=True, coerce=True),
            "revision_frequency": pa.Column(
                str,
                checks=pa.Check.isin(["Daily", "Few times a week", "Never", "Rarely"]),
                nullable=True,
                coerce=True,
            ),
            "focus_duration": pa.Column(
                str,
                checks=pa.Check.isin(["1–2 hours", "30–60 minutes", "More than 2 hours"]),
                nullable=True,
                coerce=True,
            ),
            "screen_time_non_study": pa.Column(
                str,
                checks=pa.Check.isin(["2–4 hours", "4–6 hours", "More than 6 hours"]),
                nullable=True,
                coerce=True,
            ),
            "main_distractor": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Gaming",
                        "Other",
                        "Social interactions",
                        "Social media",
                        "Video content (YouTube/OTT)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "study_consistency": pa.Column(
                str,
                checks=pa.Check.isin(["Mostly consistent", "Rarely", "Sometimes"]),
                nullable=True,
                coerce=True,
            ),
            "tasks_on_time": pa.Column(
                str,
                checks=pa.Check.isin(["Always", "Often", "Rarely", "Sometimes"]),
                nullable=True,
                coerce=True,
            ),
            "preparation_status": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Actively preparing for a goal (placements/exams)",
                        "Planning to start soon",
                        "Thinking about it",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "career_goal_clarity": pa.Column(
                str,
                checks=pa.Check.isin(["Not clear", "Somewhat clear", "Very clear"]),
                nullable=True,
                coerce=True,
            ),
            "skills_developing": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Both hard and soft skills",
                        "Hard skills (programming, data analytics, technical skills)",
                        "Soft skills (communication, teamwork, leadership, financial literacy)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "energy_level": pa.Column(float, nullable=True, coerce=True),
            "stress_level": pa.Column(float, nullable=True, coerce=True),
            "routine_rating": pa.Column(float, nullable=True, coerce=True),
            "sleepy_during_study": pa.Column(
                str,
                checks=pa.Check.isin(["Always", "Never", "Often", "Sometimes"]),
                nullable=True,
                coerce=True,
            ),
            "sleep_hours": pa.Column(
                str,
                checks=pa.Check.isin(["4–5 hours", "6–7 hours", "More than 8 hours"]),
                nullable=True,
                coerce=True,
            ),
            "career_interest": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "AI / ML",
                        "Automation Engineer",
                        "Cyber Security Analyst",
                        "Data Analyst",
                        "Digital Marketing Specialist",
                        "Other",
                        "Software Developer",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "online_courses": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "No, not interested",
                        "Not currently, but intend to in the future",
                        "Planning to enroll soon",
                        "Yes, currently enrolled in one or more courses/certifications",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "projects_internships": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Not currently, but intend to in the future",
                        "Planning to start a project/internship soon",
                        "Yes, actively working on projects/internship",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "programming_foundation": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Basic knowledge, learning while practicing",
                        "Limited knowledge, theoretical only",
                        "Strong foundation in core concepts",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "events_participation": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Never participate in such events",
                        "Occasionally participate in events",
                        "Rarely participate, mostly observe",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "assignments_on_time": pa.Column(
                str,
                checks=pa.Check.isin(["Always", "Often", "Rarely", "Sometimes"]),
                nullable=True,
                coerce=True,
            ),
            "attendance_percentage": pa.Column(
                str,
                checks=pa.Check.isin(
                    ["50% – 65%", "66% – 75%", "76% – 85%", "Above 85%", "Less than 50%"]
                ),
                nullable=True,
                coerce=True,
            ),
            "strongest_asset": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Creative/Design Skills (Innovation, UI/UX, Content)",
                        "Management/Execution (Planning, Organizing, Discipline)",
                        "Soft Skills (Communication, Leadership, Teamwork)",
                        "Technical/Hard Skills (Coding, Math, Logic)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "internal_barrier": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Difficulty with Focus / Concentration",
                        "Lack of Consistency or Determination (Difficulty sticking to a plan)",
                        "Poor Time Management / Over-scheduling",
                        "Procrastination / Low Motivation",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "external_resources": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "Never (Unaware or Not interested)",
                        "Occasionally (When needed)",
                        "Rarely (Passive)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "external_pressure": pa.Column(
                str,
                checks=pa.Check.isin(
                    [
                        "High Impact (Frequent disruption)",
                        "Low Impact (Rarely affects study)",
                        "Moderate Impact (Occasional disruption)",
                        "No Impact (Fully supportive environment)",
                    ]
                ),
                nullable=True,
                coerce=True,
            ),
            "performance_risk_level": pa.Column(
                str,
                checks=pa.Check.isin(["High Risk", "Low Risk", "Moderate Risk"]),
                nullable=True,
                coerce=True,
            ),
        },
        coerce=True,
        strict=True,
    )
