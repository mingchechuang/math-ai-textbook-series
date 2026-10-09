import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// 已實測的gb10-2 vLLM相容性：無工具時省略tools，而不是送tools:[]。
// 不記錄payload或headers，不修改其他模型，不更動遠端服務。
export function omitEmptyTools(payload: unknown, provider: string | undefined): unknown {
  if (provider !== "gb10-2-vllm" || !payload || typeof payload !== "object") return payload;
  const value = payload as Record<string, unknown>;
  if (!Array.isArray(value.tools) || value.tools.length !== 0) return payload;
  const result = { ...value };
  delete result.tools;
  delete result.tool_choice;
  delete result.parallel_tool_calls;
  return result;
}

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", (event, ctx) => omitEmptyTools(event.payload, ctx.model?.provider));
}
