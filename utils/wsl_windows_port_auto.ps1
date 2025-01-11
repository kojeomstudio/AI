# ifconfig 명령어를 찾지 못한다면 아래 명령어 실행.
# sudo apt update
# sudo apt install net-tools

$remoteport = bash.exe -c "ifconfig eth0 | grep 'inet '"
$found = $remoteport -match '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}';

if( $found ){
  $remoteport = $matches[0];
} else {
  echo "The Script Exited, the IP address of WSL 2 cannot be found";
  exit;
}

# [Ports]
# All the ports you want to forward separated by comma
$ports = @(7000, 21);

# [Static IP]
# You can change the addr to your IP config to listen to a specific address
$addr = '0.0.0.0';
$ports_a = $ports -join ",";

# Remove Firewall Exception Rules
iex "Remove-NetFireWallRule -DisplayName 'WSL 2 Firewall Unlock'";

# Adding Exception Rules for inbound and outbound Rules
iex "New-NetFireWallRule -DisplayName 'WSL 2 Firewall Unlock' -Direction Outbound -LocalPort $ports_a -Action Allow -Protocol TCP";
iex "New-NetFireWallRule -DisplayName 'WSL 2 Firewall Unlock' -Direction Inbound -LocalPort $ports_a -Action Allow -Protocol TCP";

# Set up port forwarding for each port
for( $i = 0; $i -lt $ports.length; $i++ ) {
  $port = $ports[$i];
  iex "netsh interface portproxy delete v4tov4 listenport=$port listenaddress=$addr";
  iex "netsh interface portproxy add v4tov4 listenport=$port listenaddress=$addr connectport=$port connectaddress=$remoteport";
}
