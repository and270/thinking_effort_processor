import tkinter as tk
import math
import random
from dataclasses import dataclass
import time

WIDTH, HEIGHT = 800, 800
CANVAS_CENTER = (WIDTH // 2, HEIGHT // 2)
RADIUS_HEPTAGON = 300  # Radius of heptagon vertices from center
BALL_RADIUS = 15        # Radius of each ball
GRAVITY = 100           # Acceleration due to gravity per second
FRICTION = 0.02         # Velocity damping coefficient (per second)
SPIN_FRICTION = 0.01    # Spin damping coefficient (per second)
ELASTICITY = 0.9        # Coefficient of restitution for collisions

# Colors for balls
COLORS = ['#f8b862', '#f6ad49', '#f39800', '#f08300', '#ec6d51', '#ee7948', '#ed6d3d', '#ec6800', '#ec6800', '#ee7800',
          '#eb6238', '#ea5506', '#ea5506', '#eb6101', '#e49e61', '#e45e32', '#e17b34', '#dd7a56', '#db8449', '#d66a35']

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: str
    number: int
    spin: float
    rotation_angle: float = 0.0

def main():
    root = tk.Tk()
    root.title("Bouncing Balls in Rotating Heptagon")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
    canvas.pack()

    # Initialize balls
    balls = []
    for i in range(20):
        color = COLORS[i % len(COLORS)]
        number = i + 1
        # Random initial position near center (0,0)
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, 30)  # Small radius around center
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        vx = 0
        vy = 0
        balls.append(Ball(x, y, vx, vy, BALL_RADIUS, color, number, 0.0))

    # Heptagon parameters
    num_sides = 7
    rotation_angle = 0.0
    angular_velocity = 2 * math.pi / 5  # radians per second (360 degrees in 5 seconds)

    # Precompute initial vertices of the heptagon
    initial_vertices = []
    angle_step = 2 * math.pi / num_sides
    for i in range(num_sides):
        angle = i * angle_step
        x = RADIUS_HEPTAGON * math.cos(angle)
        y = RADIUS_HEPTAGON * math.sin(angle)
        initial_vertices.append((x, y))

    # Variables for timing
    last_time = time.time()

    def to_screen(x, y):
        return x + CANVAS_CENTER[0], y + CANVAS_CENTER[1]

    def closest_point_on_segment(p, a, b):
        ap = (p[0] - a[0], p[1] - a[1])
        ab = (b[0] - a[0], b[1] - a[1])
        dot_ap_ab = ap[0] * ab[0] + ap[1] * ab[1]
        dot_ab_ab = ab[0] ** 2 + ab[1] ** 2
        if dot_ab_ab == 0:
            return a
        t = dot_ap_ab / dot_ab_ab
        if t < 0:
            return a
        elif t > 1:
            return b
        else:
            x = a[0] + t * ab[0]
            y = a[1] + t * ab[1]
            return (x, y)

    def update():
        nonlocal last_time, rotation_angle
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        # Update rotation
        rotation_angle += angular_velocity * delta_time
        rotation_angle %= (2 * math.pi)

        # Compute current polygon vertices once per frame
        current_vertices = []
        for (x, y) in initial_vertices:
            x_rot = x * math.cos(rotation_angle) - y * math.sin(rotation_angle)
            y_rot = x * math.sin(rotation_angle) + y * math.cos(rotation_angle)
            current_vertices.append((x_rot, y_rot))

        edges = []
        for i in range(num_sides):
            a = current_vertices[i]
            b = current_vertices[(i + 1) % num_sides]
            edges.append((a, b))

        # Update each ball's physics
        for ball in balls:
            # Apply gravity
            ball.vy += GRAVITY * delta_time

            # Apply linear friction (damping)
            ball.vx *= (1 - FRICTION * delta_time)
            ball.vy *= (1 - FRICTION * delta_time)

            # Update position
            ball.x += ball.vx * delta_time
            ball.y += ball.vy * delta_time

            # Check for polygon collision
            min_distance = float('inf')
            closest_edge = None
            closest_normal = None
            closest_point = None
            for edge in edges:
                a, b = edge
                p = (ball.x, ball.y)
                cp = closest_point_on_segment(p, a, b)
                distance = math.hypot(cp[0] - ball.x, cp[1] - ball.y)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = cp
                    closest_edge = edge
                    # compute normal vector
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    normal_x = dy
                    normal_y = -dx
                    length = math.hypot(normal_x, normal_y)
                    if length == 0:
                        normal_unit = (0, 0)
                    else:
                        normal_unit = (normal_x / length, normal_y / length)
                    closest_normal = normal_unit

            if min_distance < ball.radius:
                # Resolve collision with polygon edge
                normal_unit = closest_normal
                # Compute overlap
                overlap = ball.radius - min_distance
                # Move the ball back along the inward direction
                direction = (-normal_unit[0], -normal_unit[1])
                ball.x += direction[0] * overlap
                ball.y += direction[1] * overlap

                # Compute velocity reflection
                v_dot_n = ball.vx * normal_unit[0] + ball.vy * normal_unit[1]
                new_vn = -ELASTICITY * v_dot_n  # Reflect with elasticity

                # Tangential components
                vt_x = ball.vx - v_dot_n * normal_unit[0]
                vt_y = ball.vy - v_dot_n * normal_unit[1]

                # Update velocity
                ball.vx = vt_x + new_vn * normal_unit[0]
                ball.vy = vt_y + new_vn * normal_unit[1]

                # Update spin based on tangential velocity
                # Tangential velocity magnitude
                v_tangent = math.hypot(vt_x, vt_y)
                ball.spin = v_tangent / ball.radius  # Angular velocity

            # Apply spin friction
            ball.spin *= (1 - SPIN_FRICTION * delta_time)
            ball.rotation_angle += ball.spin * delta_time

        # Check ball-ball collisions
        collision_pairs = []
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                ball1 = balls[i]
                ball2 = balls[j]
                dx = ball1.x - ball2.x
                dy = ball1.y - ball2.y
                distance = math.hypot(dx, dy)
                if distance < 2 * BALL_RADIUS + 1e-5:
                    collision_pairs.append((ball1, ball2))

        # Resolve collisions between balls
        for ball1, ball2 in collision_pairs:
            dx = ball1.x - ball2.x
            dy = ball1.y - ball2.y
            distance = math.hypot(dx, dy)
            if distance < 1e-5:
                continue
            nx = dx / distance
            ny = dy / distance
            rvx = ball1.vx - ball2.vx
            rvy = ball1.vy - ball2.vy
            dot_product = rvx * nx + rvy * ny
            if dot_product > 0:
                continue  # Moving away, no collision
            # Compute impulse
            J = (2 * dot_product) / 2  # Both masses same (assuming mass=1)
            # Update velocities
            ball1.vx -= J * nx
            ball1.vy -= J * ny
            ball2.vx += J * nx
            ball2.vy += J * ny

        # Redraw everything
        canvas.delete('all')
        # Draw polygon
        current_vertices_screen = [to_screen(x, y) for (x, y) in current_vertices]
        canvas.create_polygon(current_vertices_screen, outline='black', fill='', width=2)
        # Draw balls
        for ball in balls:
            x, y = to_screen(ball.x, ball.y)
            color = ball.color
            canvas.create_oval(x - BALL_RADIUS, y - BALL_RADIUS,
                              x + BALL_RADIUS, y + BALL_RADIUS, fill=color)
            # Draw the number rotated by rotation_angle
            try:
                canvas.create_text(x, y, text=str(ball.number),
 fill='black', angle=ball.rotation_angle)
            except:
                canvas.create_text(x, y, text=str(ball.number), fill='black')

        root.after(16, update)

    root.after(0, update)
    root.mainloop()

if __name__ == '__main__':
    main()