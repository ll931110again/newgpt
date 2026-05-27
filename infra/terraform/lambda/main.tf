provider "lambdalabs" {
  api_key = var.lambdalabs_api_key != "" ? var.lambdalabs_api_key : null
}

resource "lambdalabs_instance" "pretrain" {
  region_name        = var.region_name
  instance_type_name = var.instance_type_name
  ssh_key_names      = var.ssh_key_names
  name               = var.instance_name
  file_system_names  = length(var.file_system_names) > 0 ? var.file_system_names : null

  timeouts {
    create = "30m"
  }
}
