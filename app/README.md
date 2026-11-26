# 本地微信小程序

在仓库根目录运行后端。核查文章前，按 [环境变量模板](../.env.example) 配置模型，并按 [项目说明](../README.md) 准备检索所需的数据。

```bash
python -m pip install -r requirements.txt -r app/api/requirements.txt
python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
curl http://localhost:8000/api/health
```

健康检查返回 `{"status":"ok"}`。在微信开发者工具中导入 `app/miniprogram`，选择测试号。本地模式向 `http://localhost:8000` 请求。

| 接口 | 作用 |
| --- | --- |
| `GET /api/health` | 检查后端状态 |
| `POST /api/check` | 提交文章 URL 或正文，返回证据、行为风险和处置档位 |
