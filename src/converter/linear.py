import numpy as np
from sklearn.linear_model import LogisticRegression
from src.converter.base import BaseConverter


class LinearConverter(BaseConverter):

    def convert_to_c(self, func_name: str = "predict_model") -> str:
        coef = np.asarray(self.model.coef_)
        intercept = np.asarray(self.model.intercept_)

        has_scaler = self.scaler is not None

        if coef.ndim == 1:
            return self._emit_regression_or_binary(func_name, coef, intercept, has_scaler)

        if coef.ndim == 2:  # MULTICLASS
            return self._emit_multiclass(func_name, coef, intercept, has_scaler)

        raise ValueError("Unexpected coef_ shape")

    # Regression or binary logistic
    def _emit_regression_or_binary(self, func_name, coef, intercept, has_scaler):
        c = []
        c.append("// Auto-generated linear-model C inference")
        c.append("#pragma once\n")
        
        if has_scaler:
            if hasattr(self.scaler, "mean_"):
                means = ", ".join(f"{m:.10f}f" for m in self.scaler.mean_)
                vars_ = ", ".join(f"{v:.10f}f" for v in self.scaler.scale_)
                c.append(f"static const float SCALER_MEAN[] = {{ {means} }};")
                c.append(f"static const float SCALER_SCALE[] = {{ {vars_} }};")
            else:
                raise ValueError("Scaler not supported yet.")

        coef_c = ", ".join(f"{w:.10f}f" for w in coef)
        c.append(f"static const float WEIGHTS[] = {{ {coef_c} }};")
        c.append(f"static const float BIAS = {intercept:.10f}f;\n")

        c.append(f"float {func_name}(const float *x, int n_features) {{")

        # scaler if present
        if has_scaler:
            c.append("    float x_scaled[n_features];")
            c.append("    for (int i = 0; i < n_features; i++) {")
            c.append("        x_scaled[i] = (x[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];")
            c.append("    }")
            xref = "x_scaled"
        else:
            xref = "x"

        c.append("    float s = BIAS;")
        c.append("    for (int i = 0; i < n_features; i++) {")
        c.append(f"        s += WEIGHTS[i] * {xref}[i];")
        c.append("    }")

        # Binary logistic
        if isinstance(self.model, LogisticRegression):
            c.append("    float out = 1.0f / (1.0f + expf(-s));")
            c.append("    return out;")
        else:
            c.append("    return s;")

        c.append("}")
        return "\n".join(c)

    # Multiclass
    def _emit_multiclass(self, func_name, coef, intercept, has_scaler):
        C, F = coef.shape

        c = []
        c.append("// Auto-generated multiclass logistic C inference")
        c.append("#pragma once\n")
        c.append("#include <math.h>\n")

        if has_scaler:
            means = ", ".join(f"{m:.10f}f" for m in self.scaler.mean_)
            scales = ", ".join(f"{s:.10f}f" for s in self.scaler.scale_)
            c.append(f"static const float SCALER_MEAN[] = {{ {means} }};")
            c.append(f"static const float SCALER_SCALE[] = {{ {scales} }};")

        for cls in range(C):
            w = ", ".join(f"{x:.10f}f" for x in coef[cls])
            c.append(f"static const float W_{cls}[] = {{ {w} }};")
            c.append(f"static const float B_{cls} = {intercept[cls]:.10f}f;")

        c.append("")
        c.append(f"int {func_name}(const float *x, int n_features) {{")

        if has_scaler:
            c.append("    float x_scaled[n_features];")
            c.append("    for (int i = 0; i < n_features; i++)")
            c.append("        x_scaled[i] = (x[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];")
            xref = "x_scaled"
        else:
            xref = "x"

        # softmax scores
        c.append(f"    float scores[{C}];")
        for cls in range(C):
            c.append(f"    scores[{cls}] = B_{cls};")
            c.append(f"    for (int i = 0; i < n_features; i++)")
            c.append(f"        scores[{cls}] += W_{cls}[i] * {xref}[i];")

        # softmax + argmax
        c.append("    float max_s = scores[0];")
        c.append(f"    int max_i = 0;")
        c.append(f"    for (int c = 1; c < {C}; c++) {{")
        c.append("        if (scores[c] > max_s) { max_s = scores[c]; max_i = c; }")
        c.append("    }")
        c.append("    return max_i;")
        c.append("}")

        return "\n".join(c)
