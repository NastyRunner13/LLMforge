/**
 * LLM Forge — Shared Types & Constants
 *
 * This package contains types and constants shared across the monorepo.
 * These mirror the backend Pydantic models and database schema to ensure
 * type consistency between frontend and backend.
 */

// ============================
// Enums
// ============================

export type ProjectStatus = 'active' | 'training' | 'ready' | 'failed' | 'archived';

export type DatasetStatus = 'uploading' | 'processing' | 'cleaning' | 'ready' | 'failed';

export type RunStatus =
    | 'queued'
    | 'provisioning'
    | 'downloading'
    | 'training'
    | 'saving'
    | 'paused'
    | 'completed'
    | 'failed'
    | 'cancelled';

export type EndpointStatus = 'starting' | 'running' | 'stopping' | 'stopped' | 'failed';

export type TrainingMethod = 'sft' | 'lora' | 'qlora';

export type MixedPrecision = 'bf16' | 'fp16' | 'fp32';

export type CreditType = 'purchase' | 'subscription_grant' | 'training_consume' | 'inference_consume' | 'refund';

export type UserPlan = 'free' | 'pro' | 'team';

export type WorkspaceRole = 'owner' | 'editor' | 'viewer';

// ============================
// Base Model Registry
// ============================

export interface BaseModelInfo {
    id: string;
    name: string;
    organization: string;
    huggingface_id: string;
    param_count: number;       // in billions (e.g., 7 for 7B)
    license: string;
    recommended_vram_gb: number;
    description: string;
    default_lora_r: number;
    default_lora_alpha: number;
    default_lora_targets: string[];
    context_length: number;
}

export const BASE_MODELS: BaseModelInfo[] = [
    {
        id: 'llama-3-8b',
        name: 'LLaMA 3 8B',
        organization: 'Meta',
        huggingface_id: 'meta-llama/Meta-Llama-3-8B',
        param_count: 8,
        license: 'Llama 3 Community License',
        recommended_vram_gb: 24,
        description: 'Meta\'s latest open-weight model. Strong general performance with excellent instruction following.',
        default_lora_r: 16,
        default_lora_alpha: 32,
        default_lora_targets: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        context_length: 8192,
    },
    {
        id: 'mistral-7b',
        name: 'Mistral 7B',
        organization: 'Mistral AI',
        huggingface_id: 'mistralai/Mistral-7B-v0.3',
        param_count: 7,
        license: 'Apache 2.0',
        recommended_vram_gb: 20,
        description: 'Efficient 7B model with sliding window attention. Great for general-purpose fine-tuning.',
        default_lora_r: 16,
        default_lora_alpha: 32,
        default_lora_targets: ['q_proj', 'v_proj'],
        context_length: 32768,
    },
    {
        id: 'phi-3-mini',
        name: 'Phi-3 Mini 3.8B',
        organization: 'Microsoft',
        huggingface_id: 'microsoft/Phi-3-mini-4k-instruct',
        param_count: 3.8,
        license: 'MIT',
        recommended_vram_gb: 12,
        description: 'Compact but powerful model. Punches above its weight on reasoning tasks.',
        default_lora_r: 8,
        default_lora_alpha: 16,
        default_lora_targets: ['q_proj', 'v_proj'],
        context_length: 4096,
    },
    {
        id: 'qwen-2-7b',
        name: 'Qwen 2 7B',
        organization: 'Alibaba',
        huggingface_id: 'Qwen/Qwen2-7B',
        param_count: 7,
        license: 'Apache 2.0',
        recommended_vram_gb: 20,
        description: 'Strong multilingual model with excellent code and math capabilities.',
        default_lora_r: 16,
        default_lora_alpha: 32,
        default_lora_targets: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        context_length: 32768,
    },
    {
        id: 'gemma-2-9b',
        name: 'Gemma 2 9B',
        organization: 'Google',
        huggingface_id: 'google/gemma-2-9b',
        param_count: 9,
        license: 'Gemma License',
        recommended_vram_gb: 28,
        description: 'Google\'s open model with strong benchmark performance across all categories.',
        default_lora_r: 16,
        default_lora_alpha: 32,
        default_lora_targets: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        context_length: 8192,
    },
    {
        id: 'tinyllama-1.1b',
        name: 'TinyLlama 1.1B',
        organization: 'TinyLlama',
        huggingface_id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        param_count: 1.1,
        license: 'Apache 2.0',
        recommended_vram_gb: 6,
        description: 'Tiny but capable model. Perfect for experimentation and testing pipelines.',
        default_lora_r: 8,
        default_lora_alpha: 16,
        default_lora_targets: ['q_proj', 'v_proj'],
        context_length: 2048,
    },
];

// ============================
// GPU Types & Pricing
// ============================

export interface GpuType {
    id: string;
    name: string;
    vram_gb: number;
    credits_per_hour: number;
    recommended_for: string;
}

export const GPU_TYPES: GpuType[] = [
    { id: 'T4', name: 'NVIDIA T4', vram_gb: 16, credits_per_hour: 5, recommended_for: 'Small models (< 3B params)' },
    { id: 'L4', name: 'NVIDIA L4', vram_gb: 24, credits_per_hour: 8, recommended_for: 'Medium models (3-7B params)' },
    { id: 'A10G', name: 'NVIDIA A10G', vram_gb: 24, credits_per_hour: 10, recommended_for: 'Medium models with QLoRA' },
    { id: 'A100_40GB', name: 'NVIDIA A100 40GB', vram_gb: 40, credits_per_hour: 20, recommended_for: 'Large models (7-13B params)' },
    { id: 'A100_80GB', name: 'NVIDIA A100 80GB', vram_gb: 80, credits_per_hour: 35, recommended_for: 'Full fine-tuning of 7B+ models' },
];

// ============================
// API Error Format
// ============================

export interface ApiError {
    error: {
        code: string;
        message: string;
        details?: Record<string, unknown>;
        request_id?: string;
    };
}

// ============================
// Plan Limits
// ============================

export const PLAN_LIMITS: Record<UserPlan, {
    storage_gb: number;
    gpu_minutes_per_month: number;
    concurrent_training_jobs: number;
    active_endpoints: number;
    max_file_size_gb: number;
}> = {
    free: {
        storage_gb: 5,
        gpu_minutes_per_month: 100,
        concurrent_training_jobs: 2,
        active_endpoints: 1,
        max_file_size_gb: 5,
    },
    pro: {
        storage_gb: 50,
        gpu_minutes_per_month: 1000,
        concurrent_training_jobs: 5,
        active_endpoints: 5,
        max_file_size_gb: 20,
    },
    team: {
        storage_gb: 200,
        gpu_minutes_per_month: 5000,
        concurrent_training_jobs: 20,
        active_endpoints: 20,
        max_file_size_gb: 50,
    },
};
