variable "lambdalabs_api_key" {
  description = "Lambda Cloud API key. Prefer LAMBDALABS_API_KEY env var (or LAMBDA_API_KEY via wrapper scripts)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "region_name" {
  description = "Lambda region with capacity for the instance type."
  type        = string
  default     = "us-east-1"
}

variable "instance_type_name" {
  description = "Lambda instance type (e.g. gpu_1x_a100_sxm4, gpu_1x_h100_sxm5)."
  type        = string
  default     = "gpu_1x_a100_sxm4"
}

variable "ssh_key_names" {
  description = "SSH key names already registered in Lambda Cloud."
  type        = list(string)
}

variable "instance_name" {
  description = "Display name for the GPU instance."
  type        = string
  default     = "pretrainer-pretrain"
}

variable "file_system_names" {
  description = "Optional Lambda filesystems to attach."
  type        = list(string)
  default     = []
}
