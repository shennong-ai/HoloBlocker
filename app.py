# 在 app.py 中，点击按钮后的检测部分修改如下：

is_safe, report = blocker.check(user_input, conf_threshold=0.3)  # 可调阈值
energy = report["energy"]
confidence = report["confidence"]

st.subheader("📊 检测结果")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("语义置信度", f"{confidence:.3f}")
with col_m2:
    st.metric("狄利克雷能量", f"{energy:.4f}")
with col_m3:
    if is_safe:
        st.success("✅ 安全 — 允许执行")
    else:
        st.error("🚨 断路触发 — 动作被阻止")
        st.info(f"原因: {report['trigger_reason']}")