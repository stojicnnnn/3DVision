#version 330 core

in vec3 vColor;
out vec4 FragColor;

void main() {
    // Round point shape
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;

    FragColor = vec4(vColor, 1.0);
}
