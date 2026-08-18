from pymavlink import mavutil
import time
# Thử gửi trực tiếp bằng udpout
conn = mavutil.mavlink_connection("udpout:172.22.226.176:14556", source_system=255, source_component=190)
for i in range(100):
    conn.mav.heartbeat_send(6, 8, 0, 0, 0)
    conn.mav.manual_control_send(1, 0, 0, 500, 0, 0)
    print(f"Sent packet {i+1}")
    time.sleep(0.1)
print("Done")