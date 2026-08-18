from px4_follow import PX4Follower
import time

px4 = PX4Follower()

while True:

    px4.send_velocity(
        0,
        0,
        0,
        0
    )

    print("sending")

    time.sleep(0.05)