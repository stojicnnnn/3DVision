#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uVP;

out vec3 vColor;

void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vColor = aColor;

    // Point size attenuation by depth
    float dist = length((uVP * vec4(aPos, 1.0)).xyz);
    gl_PointSize = max(1.0, 5.0 / dist);
}
