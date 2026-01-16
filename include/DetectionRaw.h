#pragma once

namespace cropandweed {

struct alignas(float) DetectionRaw {
    float x, y, w, h;
    float score;
    float class_id;
    float batch_index;
    float track_id;
};

}
