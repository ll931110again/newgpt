output "instance_id" {
  description = "Lambda instance ID."
  value       = lambdalabs_instance.pretrain.id
}

output "instance_ip" {
  description = "Public IP for SSH and bootstrap."
  value       = lambdalabs_instance.pretrain.ip
}

output "ssh_command" {
  description = "Example SSH command."
  value       = "ssh ubuntu@${lambdalabs_instance.pretrain.ip}"
}

output "bootstrap_command" {
  description = "Bootstrap pretraining from repo root after apply."
  value       = "LAMBDA_INSTANCE_IP=${lambdalabs_instance.pretrain.ip} ./infra/gpu-cloud/lambda/bootstrap_remote.sh"
}
