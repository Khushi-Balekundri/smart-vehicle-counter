import streamlit as st
import cv2
import json
import requests
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Vehicle Detection & Counting")

video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
mode  = st.radio("Mode", ["zone", "line"])
model_choice = st.selectbox(
    "Model",
    ["yolov11s", "yolov11n"],
    index=0
)

def draw_on_frame(image, points, mode):
    img  = image.copy()
    draw = ImageDraw.Draw(img)

    # draw points
    for point in points:
        x, y = point
        draw.ellipse([x-5, y-5, x+5, y+5], fill="red")

    # draw zone rectangle
    if mode == "zone" and len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # draw line
    elif mode == "line" and len(points) == 2:
        draw.line([points[0], points[1]], fill="red", width=3)

    return img


if video:
    input_path = "videos/temp.mp4"
    with open(input_path, "wb") as f:
        f.write(video.read())

    # extract first frame
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    cap.release()

    # streamlit container is typically 700px wide
    DISPLAY_W     = 700
    display_h     = int(h * DISPLAY_W / w)  # maintain aspect ratio

    # resize frame to display size
    display_frame = cv2.resize(frame, (DISPLAY_W, display_h))
    pil_image     = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    # store scale factors
    st.session_state.scale_x = w / DISPLAY_W
    st.session_state.scale_y = h / display_h

    # initialize points
    if "points" not in st.session_state:
        st.session_state.points = []
    if "last_coord" not in st.session_state:
        st.session_state.last_coord = None

    if mode == "zone":
        st.write("Click top-left then bottom-right to define zone:")
    elif mode == "line":
        st.write("Click start then end point to define line:")

    display_image = draw_on_frame(pil_image, st.session_state.points, mode)
    coords = streamlit_image_coordinates(display_image)

    if coords:
        point = (coords["x"], coords["y"])
        if point != st.session_state.last_coord:
            st.session_state.last_coord = point
            if len(st.session_state.points) < 2:
                st.session_state.points.append(point)
                st.rerun()

    st.write(f"Points selected: {st.session_state.points}")

    if st.button("Clear Points"):
        st.session_state.points = []
        st.session_state.last_coord = None  # reset last coord too
        st.rerun()

    zone = None
    line = None

    if len(st.session_state.points) >= 2:
        if len(st.session_state.points) >= 2:
            scale_x = st.session_state.get("scale_x", 1.0)
            scale_y = st.session_state.get("scale_y", 1.0)
            if mode == "zone":
                x1 = int(min(st.session_state.points[0][0], st.session_state.points[1][0]) * scale_x)
                y1 = int(min(st.session_state.points[0][1], st.session_state.points[1][1]) * scale_y)
                x2 = int(max(st.session_state.points[0][0], st.session_state.points[1][0]) * scale_x)
                y2 = int(max(st.session_state.points[0][1], st.session_state.points[1][1]) * scale_y)
                zone = (x1, y1, x2, y2)
                st.write(f"Zone: {zone}")
            elif mode == "line":
                lx1 = int(st.session_state.points[0][0] * scale_x)
                ly1 = int(st.session_state.points[0][1] * scale_y)
                lx2 = int(st.session_state.points[1][0] * scale_x)
                ly2 = int(st.session_state.points[1][1] * scale_y)
                line = ((lx1, ly1), (lx2, ly2))
                st.write(f"Line: {line}")

    if st.button("Run"):
        st.session_state.points = []  # reset after run
        with st.spinner("Processing video..."):
            with open(input_path, "rb") as f:
                response = requests.post(
                    "http://api:8003/process",
                    files={"file": f},
                    data={
                    "mode": mode,
                    "model": model_choice,
                    "zone": json.dumps(list(zone)) if zone else None,
                    "line": json.dumps([list(p) for p in line]) if line else None
                }
                )
        result = response.json()
        st.write(result)
        st.success(f"Vehicle count: {result['count']}")
        st.write(result.get("class_counts", {}))
        with open(result["output_path"], "rb") as file:
            st.download_button(
                "Download Processed Video",
                file,
                file_name="processed.mp4",
                mime="video/mp4"
            )