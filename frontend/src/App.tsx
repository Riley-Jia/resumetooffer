import { FormEvent, useEffect, useMemo, useState } from "react";

type HealthState = "checking" | "online" | "offline";

type Profile = {
  name: string;
  headline: string;
  email: string;
  phone: string;
  wechat: string;
  location: string;
  summary: string;
  skills: string[];
  education: string[];
  experience: string[];
};

type ProjectInput = {
  category: string;
  title: string;
  role: string;
  start_date: string;
  end_date: string;
  description: string;
  technologies: string[];
  highlights: string[];
};

type Project = ProjectInput & {
  id: string;
};

type ProfileProjects = {
  profile: Profile;
  projects: Project[];
};

type CareerDirectionRecommendation = {
  direction: string;
  match_score: number;
  reason: string;
  related_projects: string[];
};

type CareerDirectionsSnapshot = {
  recommendations: CareerDirectionRecommendation[];
  updated_at: string;
};

type ResumeProjectSection = {
  title: string;
  role: string;
  period: string;
  description: string;
  technologies: string[];
  details: string[];
};

type GeneratedResume = {
  id: string;
  target_direction: string;
  created_at: string;
  introduction: string;
  skills: string[];
  projects: ResumeProjectSection[];
  selected_project_ids: string[];
};

type Job = {
  id: string;
  title: string;
  company: string;
  location: string;
  level: string;
  role_family: string;
  status: string;
  required_skills: string[];
  nice_to_have_skills: string[];
  description: string;
};

type JobMatchResult = {
  job: Job;
  final_score: number;
  rule_score: number;
  llm_score: number;
  skill_coverage: number;
  location_score: number;
  level_score: number;
  role_family_score: number;
  match_reason: string;
  missing_skills: string[];
  matched_skills: string[];
  retrieval_sources: string[];
};

type JobMatchResponse = {
  matches: JobMatchResult[];
  metadata_filter: Record<string, string[] | string>;
  candidate_counts: Record<string, number>;
};

type JobSkillGap = {
  job_id: string;
  title: string;
  company: string;
  missing_skills: string[];
  matched_skills: string[];
  evaluation_description: string;
};

type LearningPlanWeek = {
  week: number;
  focus: string;
  plan_type: string;
  goals: string[];
  tasks: string[];
  deliverable: string;
};

type SkillGapAnalysisResponse = {
  has_gap: boolean;
  gap_severity: string;
  gap_summary: string;
  common_missing_skills: string[];
  priority_skills: string[];
  per_job_gaps: JobSkillGap[];
  next_step_plan: LearningPlanWeek[];
  learning_plan: LearningPlanWeek[];
};

type CareerAgentGoal = {
  target_direction: string;
  locations: string[];
  levels: string[];
  role_families: string[];
  timeline_weeks: number | null;
  project_notes: string;
};

type AgentExecutionStep = {
  name: string;
  status: string;
  detail: string;
};

type CareerAgentResponse = {
  user_message: string;
  goal: CareerAgentGoal;
  execution_plan: string[];
  steps: AgentExecutionStep[];
  project_profile_preview: ProjectInput | null;
  career_directions: CareerDirectionsSnapshot;
  generated_resume: GeneratedResume | null;
  job_matches: JobMatchResponse;
  skill_gap: SkillGapAnalysisResponse;
  state: Record<string, string | number | boolean | null>;
};

type EditChangePreview = {
  target: string;
  action: string;
  before: unknown;
  after: unknown;
};

type FollowUpQuestion = {
  field: string;
  question: string;
  priority: string;
  scope: string;
};

type InformationCompletenessResult = {
  score: number;
  status: string;
  can_continue: boolean;
  missing_required: string[];
  missing_recommended: string[];
  quality_notes: string[];
  follow_up_questions: FollowUpQuestion[];
};

type ProfileProjectEditPreview = {
  message: string;
  patch: unknown;
  changes: EditChangePreview[];
  warnings: string[];
  completeness: InformationCompletenessResult;
  debug?: Record<string, unknown>;
  has_changes: boolean;
};

type ProfileProjectEditApplyResponse = {
  profile: Profile;
  projects: Project[];
  applied_changes: EditChangePreview[];
  warnings: string[];
};

type InputRouterResponse = {
  intent: string;
  content_type: string;
  route: string;
  confidence: number;
  reason: string;
  normalized_instruction: string;
  follow_up_route: string;
  follow_up_instruction: string;
};

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const emptyProfile: Profile = {
  name: "",
  headline: "",
  email: "",
  phone: "",
  wechat: "",
  location: "",
  summary: "",
  skills: [],
  education: [],
  experience: [],
};

const emptyProject: ProjectInput = {
  category: "project",
  title: "",
  role: "",
  start_date: "",
  end_date: "",
  description: "",
  technologies: [],
  highlights: [],
};

const agentProgressMessages = [
  "正在理解目标并生成执行计划...",
  "正在检查是否需要抽取新的项目经历...",
  "正在生成职业方向推荐...",
  "正在选择项目并生成新版简历...",
  "正在召回和排序岗位...",
  "正在分析 Top3 岗位技能差距...",
  "正在生成下一步计划...",
];

function splitLines(value: string) {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitCsv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatPreviewValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.join(", ") : "空";
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  if (typeof value === "string") {
    return value || "空";
  }
  return value == null ? "空" : String(value);
}

export default function App() {
  const [activeView, setActiveView] = useState<
    "profile" | "projects" | "career" | "resumes" | "jobs" | "skillGap"
  >("profile");
  const [health, setHealth] = useState<HealthState>("checking");
  const [profile, setProfile] = useState<Profile>(emptyProfile);
  const [projects, setProjects] = useState<Project[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectDraft, setProjectDraft] = useState<ProjectInput>(emptyProject);
  const [skillsText, setSkillsText] = useState("");
  const [educationText, setEducationText] = useState("");
  const [technologiesText, setTechnologiesText] = useState("");
  const [highlightsText, setHighlightsText] = useState("");
  const [projectNotesText, setProjectNotesText] = useState("");
  const [isProfiling, setIsProfiling] = useState(false);
  const [isRecommending, setIsRecommending] = useState(false);
  const [isGeneratingResume, setIsGeneratingResume] = useState(false);
  const [isMatchingJobs, setIsMatchingJobs] = useState(false);
  const [isAnalyzingSkillGap, setIsAnalyzingSkillGap] = useState(false);
  const [isRunningAgent, setIsRunningAgent] = useState(false);
  const [isPreviewingEdit, setIsPreviewingEdit] = useState(false);
  const [isApplyingEdit, setIsApplyingEdit] = useState(false);
  const [agentMessage, setAgentMessage] = useState(
    "我想三周后投 Sydney Junior Backend 岗位，帮我选项目、生成简历、推荐岗位、分析差距并安排学习计划。",
  );
  const [agentProgressText, setAgentProgressText] = useState("");
  const [agentResult, setAgentResult] = useState<CareerAgentResponse | null>(null);
  const [inputRoute, setInputRoute] = useState<InputRouterResponse | null>(null);
  const [editPreview, setEditPreview] = useState<ProfileProjectEditPreview | null>(
    null,
  );
  const [jobTargetDirection, setJobTargetDirection] = useState("Backend Developer");
  const [jobLocationsText, setJobLocationsText] = useState("北京, 上海, 深圳");
  const [jobLevelsText, setJobLevelsText] = useState("实习, 校招, 初级, 应届, 1年以内");
  const [jobRoleFamiliesText, setJobRoleFamiliesText] = useState(
    "Backend, AI Application, Graduate Software Engineer",
  );
  const [jobMatches, setJobMatches] = useState<JobMatchResult[]>([]);
  const [jobMatchCounts, setJobMatchCounts] = useState<Record<string, number>>({});
  const [skillGapAnalysis, setSkillGapAnalysis] =
    useState<SkillGapAnalysisResponse | null>(null);
  const [careerDirections, setCareerDirections] = useState<
    CareerDirectionRecommendation[]
  >([]);
  const [careerDirectionsUpdatedAt, setCareerDirectionsUpdatedAt] = useState("");
  const [generatedResume, setGeneratedResume] = useState<GeneratedResume | null>(
    null,
  );
  const [resumeVersions, setResumeVersions] = useState<GeneratedResume[]>([]);
  const [message, setMessage] = useState("Loading profile and projects...");

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  useEffect(() => {
    const controller = new AbortController();

    async function loadData() {
      try {
        const healthResponse = await fetch(`${apiBaseUrl}/health`, {
          signal: controller.signal,
        });
        if (!healthResponse.ok) {
          throw new Error(`Health check failed with ${healthResponse.status}`);
        }

        const response = await fetch(`${apiBaseUrl}/profile-projects`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Data request failed with ${response.status}`);
        }

        const data = (await response.json()) as ProfileProjects;
        setProfile(data.profile);
        setProjects(data.projects);
        setSkillsText(data.profile.skills.join(", "));
        setEducationText(data.profile.education.join("\n"));
        void loadResumeVersions();
        void loadCareerDirections();
        void loadJobs();
        setHealth("online");
        setMessage("Backend connected");
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        setHealth("offline");
        setMessage(
          error instanceof Error
            ? `Backend unavailable: ${error.message}`
            : "Backend unavailable",
        );
      }
    }

    void loadData();

    return () => controller.abort();
  }, []);

  async function loadResumeVersions() {
    const response = await fetch(`${apiBaseUrl}/resumes`);
    if (!response.ok) {
      return;
    }

    const result = (await response.json()) as GeneratedResume[];
    setResumeVersions(result);
    if (!generatedResume && result.length > 0) {
      setGeneratedResume(result[0]);
    }
  }

  async function loadCareerDirections() {
    const response = await fetch(`${apiBaseUrl}/career-directions`);
    if (!response.ok) {
      return;
    }

    const result = (await response.json()) as CareerDirectionsSnapshot;
    setCareerDirections(result.recommendations);
    setCareerDirectionsUpdatedAt(result.updated_at);
  }

  async function loadJobs() {
    const response = await fetch(`${apiBaseUrl}/jobs`);
    if (!response.ok) {
      return;
    }

    setJobs((await response.json()) as Job[]);
  }

  function updateProfileField(field: keyof Profile, value: string) {
    setProfile((currentProfile) => ({
      ...currentProfile,
      [field]: value,
    }));
  }

  function updateProjectField(field: keyof ProjectInput, value: string) {
    setProjectDraft((currentProject) => ({
      ...currentProject,
      [field]: value,
    }));
  }

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const nextProfile = {
      ...profile,
      skills: splitCsv(skillsText),
      education: splitLines(educationText),
      experience: [],
    };

    const response = await fetch(`${apiBaseUrl}/profile`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(nextProfile),
    });

    if (!response.ok) {
      setMessage(`Profile save failed with ${response.status}`);
      return;
    }

    setProfile((await response.json()) as Profile);
    setMessage("Profile saved");
  }

  async function saveProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const payload: ProjectInput = {
      ...projectDraft,
      technologies: splitCsv(technologiesText),
      highlights: splitLines(highlightsText),
    };
    const url = selectedProjectId
      ? `${apiBaseUrl}/projects/${selectedProjectId}`
      : `${apiBaseUrl}/projects`;
    const method = selectedProjectId ? "PUT" : "POST";

    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      setMessage(`Project save failed with ${response.status}`);
      return;
    }

    const savedProject = (await response.json()) as Project;
    setProjects((currentProjects) => {
      const exists = currentProjects.some(
        (project) => project.id === savedProject.id,
      );

      if (exists) {
        return currentProjects.map((project) =>
          project.id === savedProject.id ? savedProject : project,
        );
      }

      return [...currentProjects, savedProject];
    });
    setSelectedProjectId(savedProject.id);
    setProjectDraft(savedProject);
    setTechnologiesText(savedProject.technologies.join(", "));
    setHighlightsText(savedProject.highlights.join("\n"));
    setMessage(selectedProjectId ? "Project updated" : "Project created");
  }

  async function extractProjectFromNotes() {
    if (!projectNotesText.trim()) {
      setMessage("Enter project notes before extracting");
      return;
    }

    setIsProfiling(true);
    setMessage("Extracting project preview...");

    try {
      const response = await fetch(`${apiBaseUrl}/projects/profile/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: projectNotesText }),
      });

      if (!response.ok) {
        setMessage(`Project extraction failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as ProjectInput;
      setProjectDraft(result);
      setTechnologiesText(result.technologies.join(", "));
      setHighlightsText(result.highlights.join("\n"));
      setMessage("Project preview filled. Review and save it.");
    } finally {
      setIsProfiling(false);
    }
  }

  async function recommendCareerDirections() {
    setIsRecommending(true);
    setMessage("Scoring career directions...");

    try {
      const response = await fetch(`${apiBaseUrl}/career-directions/generate`, {
        method: "POST",
      });

      if (!response.ok) {
        setMessage(`Career direction scoring failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as CareerDirectionsSnapshot;
      setCareerDirections(result.recommendations);
      setCareerDirectionsUpdatedAt(result.updated_at);
      setMessage("Career directions updated");
    } finally {
      setIsRecommending(false);
    }
  }

  async function generateResume(targetDirection: string) {
    setIsGeneratingResume(true);
    setMessage(`Generating resume for ${targetDirection}...`);

    try {
      const response = await fetch(`${apiBaseUrl}/resumes/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_direction: targetDirection }),
      });

      if (!response.ok) {
        setMessage(`Resume generation failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as GeneratedResume;
      setGeneratedResume(result);
      setResumeVersions((currentVersions) => [result, ...currentVersions]);
      setMessage("Resume generated");
    } finally {
      setIsGeneratingResume(false);
    }
  }

  async function generateJobMatches() {
    setIsMatchingJobs(true);
    setMessage("Matching jobs...");

    try {
      const response = await fetch(`${apiBaseUrl}/job-matches/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_direction: jobTargetDirection,
          locations: splitCsv(jobLocationsText),
          levels: splitCsv(jobLevelsText),
          role_families: splitCsv(jobRoleFamiliesText),
          status: "active",
          top_k: 10,
          llm_candidate_count: 20,
        }),
      });

      if (!response.ok) {
        setMessage(`Job matching failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as JobMatchResponse;
      setJobMatches(result.matches);
      setJobMatchCounts(result.candidate_counts);
      setSkillGapAnalysis(null);
      setMessage("Job matches updated");
    } finally {
      setIsMatchingJobs(false);
    }
  }

  async function analyzeTop3SkillGap() {
    const topJobs = jobMatches.slice(0, 3).map((match) => match.job);
    if (topJobs.length === 0) {
      setMessage("Generate job matches before skill gap analysis");
      return;
    }

    setIsAnalyzingSkillGap(true);
    setMessage("Analyzing skill gap...");

    try {
      const response = await fetch(`${apiBaseUrl}/skill-gap/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobs: topJobs,
          top_matches: jobMatches.slice(0, 3),
          user_skills: splitCsv(skillsText),
          target_direction: jobTargetDirection,
        }),
      });

      if (!response.ok) {
        setMessage(`Skill gap analysis failed with ${response.status}`);
        return;
      }

      setSkillGapAnalysis((await response.json()) as SkillGapAnalysisResponse);
      setActiveView("skillGap");
      setMessage("Skill gap analysis updated");
    } finally {
      setIsAnalyzingSkillGap(false);
    }
  }

  async function routeInput(message: string) {
    const response = await fetch(`${apiBaseUrl}/input/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`Input routing failed with ${response.status}`);
    }

    const result = (await response.json()) as InputRouterResponse;
    setInputRoute(result);
    return result;
  }

  async function executeCareerAgent(progressStart = agentProgressMessages[0]) {
    setIsRunningAgent(true);
    setAgentProgressText(progressStart);
    setMessage("Running career agent...");
    let progressIndex = 0;
    const progressTimer = window.setInterval(() => {
      progressIndex = Math.min(progressIndex + 1, agentProgressMessages.length - 1);
      setAgentProgressText(agentProgressMessages[progressIndex]);
    }, 2500);

    try {
      const response = await fetch(`${apiBaseUrl}/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: agentMessage }),
      });

      if (!response.ok) {
        setMessage(`Career agent failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as CareerAgentResponse;
      setAgentResult(result);
      setCareerDirections(result.career_directions.recommendations);
      setCareerDirectionsUpdatedAt(result.career_directions.updated_at);
      setGeneratedResume(result.generated_resume);
      setJobMatches(result.job_matches.matches);
      setJobMatchCounts(result.job_matches.candidate_counts);
      setSkillGapAnalysis(result.skill_gap);
      setJobTargetDirection(result.goal.target_direction || jobTargetDirection);
      if (result.goal.locations.length > 0) {
        setJobLocationsText(result.goal.locations.join(", "));
      }
      if (result.goal.levels.length > 0) {
        setJobLevelsText(result.goal.levels.join(", "));
      }
      if (result.goal.role_families.length > 0) {
        setJobRoleFamiliesText(result.goal.role_families.join(", "));
      }
      if (result.project_profile_preview) {
        setProjectDraft(result.project_profile_preview);
        setTechnologiesText(result.project_profile_preview.technologies.join(", "));
        setHighlightsText(result.project_profile_preview.highlights.join("\n"));
        setActiveView("projects");
      } else {
        setActiveView("skillGap");
      }
      void loadResumeVersions();
      void loadCareerDirections();
      setAgentProgressText("Agent 已完成，结果已同步到各模块。");
      setMessage("Career agent completed");
    } finally {
      window.clearInterval(progressTimer);
      setIsRunningAgent(false);
    }
  }

  async function runCareerAgent() {
    if (!agentMessage.trim()) {
      setMessage("Enter an agent instruction first");
      return;
    }

    setAgentProgressText("正在判断输入内容和用户意图...");
    setMessage("Routing input...");

    let routed: InputRouterResponse;
    try {
      routed = await routeInput(agentMessage);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Input routing failed");
      return;
    }

    if (routed.route === "profile_project_edit_preview" && routed.confidence >= 0.65) {
      await previewProfileProjectEdit();
      return;
    }

    if (routed.route !== "career_agent_run" || routed.confidence < 0.65) {
      setAgentProgressText("需要确认输入意图，请在下方选择资料编辑、求职 Agent 或岗位 JD 分析。");
      setMessage("Input needs confirmation");
      return;
    }

    await executeCareerAgent();
  }

  async function forceEditPreview() {
    setAgentProgressText("已按资料/项目编辑处理，正在生成修改预览...");
    await previewProfileProjectEdit();
  }

  async function forceCareerAgent() {
    setAgentProgressText("已按求职 Agent 处理，正在运行完整流程...");
    await executeCareerAgent("正在按用户确认运行 Career Agent...");
  }

  function handleJobPostingAnalysisChoice() {
    setAgentProgressText("岗位 JD 分析工具尚未实现；当前不会写入资料，也不会运行求职 Agent。");
    setMessage("Job posting analysis is not implemented yet");
  }

  async function previewProfileProjectEdit() {
    if (!agentMessage.trim()) {
      setMessage("Enter an edit instruction first");
      return;
    }

    setIsPreviewingEdit(true);
    setEditPreview(null);
    setAgentProgressText("正在解析 Profile / Project 编辑意图...");
    setMessage("Previewing profile/project edit...");

    try {
      let routed: InputRouterResponse | null = inputRoute;
      try {
        routed = await routeInput(agentMessage);
      } catch {
        routed = null;
        setInputRoute(null);
      }

      const response = await fetch(`${apiBaseUrl}/profile-projects/edit/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: agentMessage,
          router_intent: routed?.intent ?? "",
          router_content_type: routed?.content_type ?? "",
          normalized_instruction: routed?.normalized_instruction ?? "",
        }),
      });

      if (!response.ok) {
        setMessage(`Edit preview failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as ProfileProjectEditPreview;
      setEditPreview(result);
      setAgentProgressText(
        result.has_changes
          ? "已生成编辑预览，请确认后再应用。"
          : "没有识别到明确的 Profile / Project 修改。",
      );
      setMessage("Edit preview ready");
    } finally {
      setIsPreviewingEdit(false);
    }
  }

  async function applyProfileProjectEdit() {
    if (!editPreview?.has_changes) {
      setMessage("No edit preview to apply");
      return;
    }

    setIsApplyingEdit(true);
    setAgentProgressText("正在应用已确认的 Profile / Project 修改...");
    setMessage("Applying profile/project edit...");

    try {
      const response = await fetch(`${apiBaseUrl}/profile-projects/edit/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patch: editPreview.patch }),
      });

      if (!response.ok) {
        setMessage(`Apply edit failed with ${response.status}`);
        return;
      }

      const result = (await response.json()) as ProfileProjectEditApplyResponse;
      setProfile(result.profile);
      setProjects(result.projects);
      setSkillsText(result.profile.skills.join(", "));
      setEducationText(result.profile.education.join("\n"));
      setEditPreview(null);
      if (inputRoute?.follow_up_route === "career_agent_run") {
        setAgentProgressText("修改已保存，正在继续运行 Career Agent...");
        await executeCareerAgent("正在基于已保存资料继续运行 Career Agent...");
        return;
      }
      setAgentProgressText("修改已应用并保存到数据库。");
      setMessage("Profile/project edit applied");
    } finally {
      setIsApplyingEdit(false);
    }
  }

  function startNewProject() {
    setSelectedProjectId(null);
    setProjectDraft(emptyProject);
    setTechnologiesText("");
    setHighlightsText("");
    setProjectNotesText("");
  }

  function editProject(project: Project) {
    setSelectedProjectId(project.id);
    setProjectDraft(project);
    setTechnologiesText(project.technologies.join(", "));
    setHighlightsText(project.highlights.join("\n"));
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <p className="eyebrow">Resume to Offer</p>
          <h1>Profile & Projects</h1>
        </div>
        <div className="connection" aria-live="polite">
          <span className={`status-dot status-dot-${health}`} />
          <span>{message}</span>
        </div>
      </header>

      <nav className="view-tabs" aria-label="Primary views">
        <button
          className={activeView === "profile" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("profile")}
        >
          个人信息
        </button>
        <button
          className={activeView === "projects" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("projects")}
        >
          项目经历
        </button>
        <button
          className={activeView === "career" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("career")}
        >
          职业方向
        </button>
        <button
          className={activeView === "resumes" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("resumes")}
        >
          简历详情
        </button>
        <button
          className={activeView === "jobs" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("jobs")}
        >
          岗位列表
        </button>
        <button
          className={activeView === "skillGap" ? "view-tab view-tab-active" : "view-tab"}
          type="button"
          onClick={() => setActiveView("skillGap")}
        >
          差距与计划
        </button>
      </nav>

      <section className="panel agent-panel">
        <div className="section-heading">
          <h2>Career Agent Orchestrator</h2>
          <div className="agent-actions">
            <button type="button" onClick={runCareerAgent} disabled={isRunningAgent}>
              {isRunningAgent ? "Running..." : "Run Agent"}
            </button>
            <button
              type="button"
              onClick={previewProfileProjectEdit}
              disabled={isPreviewingEdit || isApplyingEdit}
            >
              {isPreviewingEdit ? "Previewing..." : "Preview Edit"}
            </button>
          </div>
        </div>
        <label>
          Global Instruction
          <textarea
            value={agentMessage}
            onChange={(event) => setAgentMessage(event.target.value)}
            rows={3}
            placeholder="我想三周后投 Sydney Junior Backend 岗位，帮我选项目、生成简历、推荐岗位、分析差距并安排学习计划。"
          />
        </label>
        {agentProgressText ? (
          <p className="agent-progress" aria-live="polite">
            {agentProgressText}
          </p>
        ) : null}
        {inputRoute ? (
          <section className="input-route">
            <strong>Input Router</strong>
            <span>{inputRoute.route}</span>
            <span>Confidence {Math.round(inputRoute.confidence * 100)}%</span>
            <p>
              {inputRoute.intent} · {inputRoute.content_type}
              {inputRoute.follow_up_route ? ` · then ${inputRoute.follow_up_route}` : ""}
            </p>
            <p>{inputRoute.reason}</p>
            {(inputRoute.route === "need_confirmation" ||
              inputRoute.confidence < 0.65) ? (
              <div className="confirmation-actions">
                <button type="button" onClick={forceEditPreview}>
                  按资料编辑处理
                </button>
                <button type="button" onClick={forceCareerAgent}>
                  按求职 Agent 处理
                </button>
                <button type="button" onClick={handleJobPostingAnalysisChoice}>
                  按岗位 JD 分析处理
                </button>
              </div>
            ) : null}
          </section>
        ) : null}
        {editPreview ? (
          <section className="edit-preview">
            <div className="section-heading">
              <h3>Profile & Project Edit Preview</h3>
              <button
                type="button"
                onClick={applyProfileProjectEdit}
                disabled={!editPreview.has_changes || isApplyingEdit}
              >
                {isApplyingEdit ? "Applying..." : "Apply Changes"}
              </button>
            </div>
            {editPreview.warnings.length > 0 ? (
              <div className="edit-warnings">
                {editPreview.warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>
            ) : null}
            {editPreview.completeness ? (
              <div className="completeness-panel">
                <div className="completeness-header">
                  <strong>Information Completeness</strong>
                  <span>{editPreview.completeness.score}%</span>
                  <span>{editPreview.completeness.status}</span>
                </div>
                {editPreview.completeness.missing_required.length > 0 ? (
                  <div>
                    <small>Required Missing</small>
                    <div className="tag-list">
                      {editPreview.completeness.missing_required.map((field) => (
                        <span key={field}>{field}</span>
                      ))}
                    </div>
                  </div>
                ) : null}
                {editPreview.completeness.missing_recommended.length > 0 ? (
                  <div>
                    <small>Recommended Missing</small>
                    <div className="tag-list">
                      {editPreview.completeness.missing_recommended.map((field) => (
                        <span key={field}>{field}</span>
                      ))}
                    </div>
                  </div>
                ) : null}
                {editPreview.completeness.quality_notes.length > 0 ? (
                  <div>
                    <small>Quality Notes</small>
                    {editPreview.completeness.quality_notes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                ) : null}
                {editPreview.completeness.follow_up_questions.length > 0 ? (
                  <div>
                    <small>Follow-up Questions</small>
                    <ol className="follow-up-list">
                      {editPreview.completeness.follow_up_questions.map((question) => (
                        <li key={`${question.scope}-${question.field}`}>
                          <span>{question.priority}</span>
                          {question.question}
                        </li>
                      ))}
                    </ol>
                  </div>
                ) : null}
              </div>
            ) : null}
            {editPreview.debug ? (
              <details className="debug-panel">
                <summary>Preview Debug</summary>
                <pre>{JSON.stringify(editPreview.debug, null, 2)}</pre>
              </details>
            ) : null}
            {editPreview.changes.length === 0 ? (
              <p className="empty-state">没有可确认的修改。</p>
            ) : (
              <div className="edit-change-list">
                {editPreview.changes.map((change, index) => (
                  <article className="edit-change" key={`${change.target}-${index}`}>
                    <strong>{change.target}</strong>
                    <span>{change.action}</span>
                    <div>
                      <small>Before</small>
                      <p>{formatPreviewValue(change.before)}</p>
                    </div>
                    <div>
                      <small>After</small>
                      <p>{formatPreviewValue(change.after)}</p>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        ) : null}
        {agentResult ? (
          <div className="agent-result">
            <ol className="agent-steps">
              {agentResult.steps.map((step, index) => (
                <li
                  className={`agent-step agent-step-${step.status}`}
                  key={`${step.name}-${step.status}`}
                >
                  <span className="agent-step-index">{index + 1}</span>
                  <strong>{step.name}</strong>
                  <span className="agent-step-status">{step.status}</span>
                  <p>{step.detail}</p>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </section>

      {activeView === "profile" ? (
      <section className="profile-view">
        <form className="panel profile-panel" onSubmit={saveProfile}>
          <div className="section-heading">
            <h2>Profile</h2>
            <button type="submit">Save Profile</button>
          </div>

          <label>
            Name
            <input
              value={profile.name}
              onChange={(event) => updateProfileField("name", event.target.value)}
              placeholder="Jane Chen"
            />
          </label>

          <label>
            Headline
            <input
              value={profile.headline}
              onChange={(event) =>
                updateProfileField("headline", event.target.value)
              }
              placeholder="Full-stack developer"
            />
          </label>

          <div className="field-row">
            <label>
              Email
              <input
                value={profile.email}
                onChange={(event) =>
                  updateProfileField("email", event.target.value)
                }
                placeholder="jane@example.com"
              />
            </label>
            <label>
              Phone
              <input
                value={profile.phone}
                onChange={(event) =>
                  updateProfileField("phone", event.target.value)
                }
                placeholder="+61 400 000 000"
              />
            </label>
          </div>

          <label>
            WeChat
            <input
              value={profile.wechat}
              onChange={(event) =>
                updateProfileField("wechat", event.target.value)
              }
              placeholder="wechat-id"
            />
          </label>

          <label>
            Location
            <input
              value={profile.location}
              onChange={(event) =>
                updateProfileField("location", event.target.value)
              }
              placeholder="Sydney, Australia"
            />
          </label>

          <label>
            Summary
            <textarea
              value={profile.summary}
              onChange={(event) =>
                updateProfileField("summary", event.target.value)
              }
              placeholder="Short professional summary"
              rows={5}
            />
          </label>

          <label>
            Skills
            <input
              value={skillsText}
              onChange={(event) => setSkillsText(event.target.value)}
              placeholder="React, FastAPI, PostgreSQL"
            />
          </label>

          <label>
            Education
            <textarea
              value={educationText}
              onChange={(event) => setEducationText(event.target.value)}
              placeholder={"One education item per line\nUniversity | Degree | 2025.07 - 2026.09"}
              rows={4}
            />
          </label>

        </form>
      </section>
      ) : activeView === "projects" ? (
      <section className="projects-view">
        <section className="panel projects-panel">
          <div className="section-heading">
            <h2>Projects</h2>
            <button type="button" onClick={startNewProject}>
              New Project
            </button>
          </div>

          <div className="project-list">
            {projects.length === 0 ? (
              <p className="empty-state">No projects yet.</p>
            ) : (
              projects.map((project) => (
                <button
                  className={
                    project.id === selectedProjectId
                      ? "project-item project-item-active"
                      : "project-item"
                  }
                  key={project.id}
                  type="button"
                  onClick={() => editProject(project)}
                >
                  <strong>{project.title}</strong>
                  <span>
                    {project.category === "work_experience" ? "工作/实习" : "普通项目"} ·{" "}
                    {project.role || "No role set"}
                  </span>
                </button>
              ))
            )}
          </div>
        </section>

        <form className="panel project-form-panel" onSubmit={saveProject}>
          <div className="section-heading">
            <h2>{selectedProject ? "Edit Project" : "Add Project"}</h2>
            <button type="submit">
              {selectedProject ? "Update Project" : "Create Project"}
            </button>
          </div>

          <label>
            Category
            <select
              value={projectDraft.category}
              onChange={(event) => updateProjectField("category", event.target.value)}
            >
              <option value="project">普通项目</option>
              <option value="work_experience">工作/实习经历</option>
            </select>
          </label>

          <label>
            Title
            <input
              value={projectDraft.title}
              onChange={(event) => updateProjectField("title", event.target.value)}
              placeholder="Resume Builder"
              required
            />
          </label>

          <label>
            Role
            <input
              value={projectDraft.role}
              onChange={(event) => updateProjectField("role", event.target.value)}
              placeholder="Frontend Developer"
            />
          </label>

          <div className="field-row">
            <label>
              Start
              <input
                value={projectDraft.start_date}
                onChange={(event) =>
                  updateProjectField("start_date", event.target.value)
                }
                placeholder="2025-01"
              />
            </label>
            <label>
              End
              <input
                value={projectDraft.end_date}
                onChange={(event) =>
                  updateProjectField("end_date", event.target.value)
                }
                placeholder="2025-06"
              />
            </label>
          </div>

          <label>
            Description
            <textarea
              value={projectDraft.description}
              onChange={(event) =>
                updateProjectField("description", event.target.value)
              }
              placeholder="What the project does and your contribution"
              rows={4}
            />
          </label>

          <label>
            Technologies
            <input
              value={technologiesText}
              onChange={(event) => setTechnologiesText(event.target.value)}
              placeholder="React, TypeScript, FastAPI"
            />
          </label>

          <label>
            Highlights
            <textarea
              value={highlightsText}
              onChange={(event) => setHighlightsText(event.target.value)}
              placeholder={"One achievement per line\nReduced manual work by 40%"}
              rows={5}
            />
          </label>
        </form>
      </section>
      ) : activeView === "career" ? (
      <section className="career-view">
        <section className="panel career-panel">
          <div className="section-heading">
            <h2>Career Directions</h2>
            <button
              type="button"
              onClick={recommendCareerDirections}
              disabled={isRecommending}
            >
              {isRecommending ? "Scoring..." : "Recommend"}
            </button>
          </div>
          {careerDirectionsUpdatedAt ? (
            <p className="panel-meta">
              Updated {new Date(careerDirectionsUpdatedAt).toLocaleString()}
            </p>
          ) : null}

          <div className="direction-list">
            {careerDirections.length === 0 ? (
              <p className="empty-state">No recommendations yet.</p>
            ) : (
              careerDirections.map((item) => (
                <article className="direction-item" key={item.direction}>
                  <div className="direction-score-row">
                    <strong>{item.direction}</strong>
                    <span>{item.match_score}</span>
                  </div>
                  <p>{item.reason}</p>
                  {item.related_projects.length > 0 ? (
                    <small>{item.related_projects.join(" · ")}</small>
                  ) : null}
                  <button
                    className="direction-action"
                    type="button"
                    onClick={() => {
                      void generateResume(item.direction);
                      setActiveView("resumes");
                    }}
                    disabled={isGeneratingResume}
                  >
                    Generate Resume
                  </button>
                </article>
              ))
            )}
          </div>
        </section>
      </section>
      ) : activeView === "resumes" ? (
      <section className="resume-detail-view">
        <section className="panel resume-panel">
          <div className="section-heading">
            <h2>Resume Versions</h2>
            {generatedResume ? (
              <span className="resume-target">
                {generatedResume.target_direction}
              </span>
            ) : null}
          </div>

          <div className="resume-version-layout">
            <div className="resume-version-list">
              {resumeVersions.length === 0 ? (
                <p className="empty-state">No resume versions yet.</p>
              ) : (
                resumeVersions.map((resume, index) => (
                  <button
                    className={
                      resume.id === generatedResume?.id
                        ? "resume-version-item resume-version-item-active"
                        : "resume-version-item"
                    }
                    key={resume.id}
                    type="button"
                    onClick={() => setGeneratedResume(resume)}
                  >
                    <strong>Version {resumeVersions.length - index}</strong>
                    <span>{resume.target_direction}</span>
                    {resume.created_at ? (
                      <small>{new Date(resume.created_at).toLocaleString()}</small>
                    ) : null}
                  </button>
                ))
              )}
            </div>

            {generatedResume ? (
              <article className="resume-preview">
                <h3>个人简介</h3>
                <p>{generatedResume.introduction}</p>

                <h3>个人信息</h3>
                <p>
                  {profile.name}
                  {profile.phone ? ` | Tel: ${profile.phone}` : ""}
                  {profile.wechat ? ` | 微信: ${profile.wechat}` : ""}
                  {profile.email ? ` | 邮箱: ${profile.email}` : ""}
                </p>

                {profile.education.length > 0 ? (
                  <>
                    <h3>教育背景</h3>
                    <ul>
                      {profile.education.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </>
                ) : null}

                <h3>技能</h3>
                <p>{generatedResume.skills.join("、")}</p>

                <h3>项目经历</h3>
                {generatedResume.projects.map((project) => (
                  <section className="resume-project" key={project.title}>
                    <h4>
                      {project.title}
                      {project.role ? ` | ${project.role}` : ""}
                      {project.period ? ` | ${project.period}` : ""}
                    </h4>
                    <p>
                      <strong>项目简介：</strong>
                      {project.description}
                    </p>
                    {project.technologies.length > 0 ? (
                      <p>
                        <strong>技术栈：</strong>
                        {project.technologies.join("、")}
                      </p>
                    ) : null}
                    {project.details.length > 0 ? (
                      <>
                        <p>
                          <strong>项目细节：</strong>
                        </p>
                        <ol>
                          {project.details.map((detail) => (
                            <li key={detail}>{detail}</li>
                          ))}
                        </ol>
                      </>
                    ) : null}
                  </section>
                ))}
              </article>
            ) : null}
          </div>
        </section>
      </section>
      ) : activeView === "jobs" ? (
      <section className="jobs-view">
        <section className="panel job-match-panel">
          <div className="section-heading">
            <h2>岗位推荐</h2>
            <div className="panel-actions">
              <button
                type="button"
                onClick={generateJobMatches}
                disabled={isMatchingJobs}
              >
                {isMatchingJobs ? "Matching..." : "Generate Matches"}
              </button>
              <button
                type="button"
                onClick={analyzeTop3SkillGap}
                disabled={isAnalyzingSkillGap || jobMatches.length === 0}
              >
                {isAnalyzingSkillGap ? "Analyzing..." : "Analyze Top3 Skill Gap"}
              </button>
            </div>
          </div>

          <div className="field-row">
            <label>
              Target Direction
              <input
                value={jobTargetDirection}
                onChange={(event) => setJobTargetDirection(event.target.value)}
                placeholder="Backend Developer"
              />
            </label>
            <label>
              Cities
              <input
                value={jobLocationsText}
                onChange={(event) => setJobLocationsText(event.target.value)}
                placeholder="北京, 上海, 深圳"
              />
            </label>
          </div>

          <div className="field-row">
            <label>
              Levels
              <input
                value={jobLevelsText}
                onChange={(event) => setJobLevelsText(event.target.value)}
                placeholder="实习, 校招, 初级"
              />
            </label>
            <label>
              Role Families
              <input
                value={jobRoleFamiliesText}
                onChange={(event) => setJobRoleFamiliesText(event.target.value)}
                placeholder="Backend, AI Application"
              />
            </label>
          </div>

          {Object.keys(jobMatchCounts).length > 0 ? (
            <p className="panel-meta">
              Metadata {jobMatchCounts.metadata_filtered ?? 0} · BM25{" "}
              {jobMatchCounts.bm25_top ?? 0} · Chroma{" "}
              {jobMatchCounts.chroma_top ?? 0} · Merged{" "}
              {jobMatchCounts.merged_candidates ?? 0} · 语境评估{" "}
              {jobMatchCounts.llm_reranked ?? 0}
            </p>
          ) : null}

          <div className="match-list">
            {jobMatches.length === 0 ? (
              <p className="empty-state">No job matches generated yet.</p>
            ) : (
              jobMatches.map((match, index) => (
                <article className="match-card" key={match.job.id}>
                  <div className="job-card-header">
                    <div>
                      <h3>
                        #{index + 1} {match.job.title}
                      </h3>
                      <p>
                        {match.job.company} · {match.job.location} ·{" "}
                        {match.job.level} · {match.job.role_family}
                      </p>
                    </div>
                    <span>{Math.round(match.final_score)}</span>
                  </div>
                  <p className="job-description">{match.match_reason}</p>
                  <div className="score-grid">
                    <span>Rule {Math.round(match.rule_score)}</span>
                    <span>语境 {Math.round(match.llm_score)}</span>
                    <span>Skills {Math.round(match.skill_coverage * 100)}%</span>
                    <span>{match.retrieval_sources.join(" + ")}</span>
                  </div>
                  {match.matched_skills.length > 0 ? (
                    <div className="job-skill-row">
                      <strong>匹配技能</strong>
                      <div>
                        {match.matched_skills.map((skill) => (
                          <span className="skill-chip" key={skill}>
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {match.missing_skills.length > 0 ? (
                    <div className="job-skill-row">
                      <strong>缺失技能</strong>
                      <div>
                        {match.missing_skills.map((skill) => (
                          <span className="skill-chip skill-chip-muted" key={skill}>
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel jobs-panel">
          <div className="section-heading">
            <h2>岗位列表</h2>
            <span className="job-count">{jobs.length} jobs</span>
          </div>

          <div className="job-list">
            {jobs.length === 0 ? (
              <p className="empty-state">No jobs loaded yet.</p>
            ) : (
              jobs.map((job) => (
                <article className="job-card" key={job.id}>
                  <div className="job-card-header">
                    <div>
                      <h3>{job.title}</h3>
                      <p>
                        {job.company} · {job.location} · {job.level}
                      </p>
                    </div>
                    <span>{job.role_family}</span>
                  </div>
                  <p className="job-description">{job.description}</p>
                  <div className="job-skill-row">
                    <strong>必备</strong>
                    <div>
                      {job.required_skills.map((skill) => (
                        <span className="skill-chip" key={skill}>
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  {job.nice_to_have_skills.length > 0 ? (
                    <div className="job-skill-row">
                      <strong>加分</strong>
                      <div>
                        {job.nice_to_have_skills.map((skill) => (
                          <span className="skill-chip skill-chip-muted" key={skill}>
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </article>
              ))
            )}
          </div>
        </section>
      </section>
      ) : (
      <section className="skill-gap-view">
        {skillGapAnalysis ? (
          <section className="panel skill-gap-panel">
            <div className="section-heading">
              <h2>差距分析与下一步计划</h2>
              <span className="job-count">{skillGapAnalysis.gap_severity}</span>
            </div>

            <p className="gap-summary">{skillGapAnalysis.gap_summary}</p>

            <div className="job-skill-row">
              <strong>共同缺失技能</strong>
              <div>
                {skillGapAnalysis.common_missing_skills.length === 0 ? (
                  <span className="skill-chip">无共同缺失</span>
                ) : (
                  skillGapAnalysis.common_missing_skills.map((skill) => (
                    <span className="skill-chip skill-chip-muted" key={skill}>
                      {skill}
                    </span>
                  ))
                )}
              </div>
            </div>

            <div className="job-skill-row">
              <strong>优先补齐技能</strong>
              <div>
                {skillGapAnalysis.priority_skills.slice(0, 8).map((skill) => (
                  <span className="skill-chip" key={skill}>
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="gap-list">
              {skillGapAnalysis.per_job_gaps.map((gap) => (
                <article className="gap-card" key={gap.job_id}>
                  <h3>{gap.title}</h3>
                  <p>{gap.company}</p>
                  {gap.evaluation_description && (
                    <p className="gap-evaluation">{gap.evaluation_description}</p>
                  )}
                  <div className="job-skill-row">
                    <strong>缺失</strong>
                    <div>
                      {gap.missing_skills.length === 0 ? (
                        <span className="skill-chip">无</span>
                      ) : (
                        gap.missing_skills.map((skill) => (
                          <span className="skill-chip skill-chip-muted" key={skill}>
                            {skill}
                          </span>
                        ))
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>

            <div className="plan-list">
              {skillGapAnalysis.next_step_plan.map((week) => (
                <article className="plan-card" key={week.week}>
                  <h3>
                    Week {week.week}: {week.focus}
                  </h3>
                  <span className="plan-type">{week.plan_type}</span>
                  <strong>Goals</strong>
                  <ul>
                    {week.goals.map((goal) => (
                      <li key={goal}>{goal}</li>
                    ))}
                  </ul>
                  <strong>Tasks</strong>
                  <ul>
                    {week.tasks.map((task) => (
                      <li key={task}>{task}</li>
                    ))}
                  </ul>
                  <p>
                    <strong>Deliverable: </strong>
                    {week.deliverable}
                  </p>
                </article>
              ))}
            </div>
          </section>
        ) : (
          <section className="panel skill-gap-panel">
            <div className="section-heading">
              <h2>差距分析与下一步计划</h2>
            </div>
            <p className="empty-state">
              Generate job matches, then run Analyze Top3 Skill Gap.
            </p>
          </section>
        )}
      </section>
      )}
    </main>
  );
}
