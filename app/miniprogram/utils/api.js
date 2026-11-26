// API 工具 —— 封装与后端的通信（D53）
//
// 两种后端，由 app.js 的 globalData.backend 切换：
//   cloud  微信云托管：wx.cloud.callContainer，走微信内网，无需配置域名，自动带上用户身份（用于限次）
//   local  本机后端：wx.request 直连 apiBase
//
// 核查要 15–60 秒，云托管版后端改为「提交任务 → 轮询结果」，避免请求一直挂着被网关断开。
// 旧版本机后端（app/api/server.py）直接返回结果，这里两种都兼容。

const app = getApp()

const POLL_INTERVAL = 3000   // 每 3 秒查一次
const POLL_TIMEOUT = 180000  // 最多等 3 分钟

function request(path, method, data, timeout) {
  const g = app.globalData
  return new Promise((resolve, reject) => {
    if (g.backend === 'cloud') {
      wx.cloud.callContainer({
        config: { env: g.cloudEnv },
        path: path,
        method: method,
        timeout: timeout || 60000,     // 云托管最小实例为 0 时首个请求要冷启动，给足时间
        header: { 'X-WX-SERVICE': g.cloudService, 'content-type': 'application/json' },
        data: data,
        success(res) {
          if (res.statusCode === 200 && res.data) resolve(res.data)
          else reject(new Error('服务器返回异常（' + res.statusCode + '）'))
        },
        fail() { reject(new Error('连接云端服务失败，请稍后再试')) }
      })
    } else {
      wx.request({
        url: g.apiBase + path,
        method: method,
        header: { 'Content-Type': 'application/json' },
        data: data,
        timeout: timeout || 15000,
        success(res) {
          if (res.statusCode === 200 && res.data) resolve(res.data)
          else reject(new Error('服务器返回异常'))
        },
        fail() { reject(new Error('网络连接失败，请确认后端服务已启动')) }
      })
    }
  })
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * 核查文章
 * @param {Object} data - { url, text, title }
 * @returns {Promise<Object>} 核查结果（字段见后端 report_to_dict）
 */
async function checkArticle(data) {
  const first = await request('/api/check', 'POST', data, 120000)
  if (!first.job_id) return first            // 出错，或旧版后端直接给了结果

  const start = Date.now()
  while (Date.now() - start < POLL_TIMEOUT) {
    await sleep(POLL_INTERVAL)
    const r = await request('/api/result/' + first.job_id, 'GET')
    if (r.status === 'done') return r
    if (r.status === 'error') return { ok: false, error: r.error }
  }
  return { ok: false, error: '核查超时，请稍后重试' }
}

/**
 * 上传截图到云托管对象存储（截图核查，D54），返回 fileID 列表。
 * 只能放在 screenshots/ 目录下：后端只接受这个目录里的文件，识别完会删除。
 */
function uploadScreenshots(paths) {
  const env = app.globalData.cloudEnv
  return Promise.all(paths.map((filePath, i) => new Promise((resolve, reject) => {
    const ext = (filePath.match(/\.(\w+)$/) || [, 'jpg'])[1]
    wx.cloud.uploadFile({
      cloudPath: `screenshots/${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${i}.${ext}`,
      filePath: filePath,
      config: { env: env },
      success: res => resolve(res.fileID),
      fail: () => reject(new Error('截图上传失败，请检查网络后重试'))
    })
  })))
}

/**
 * 健康检查
 */
async function healthCheck() {
  try {
    const r = await request('/api/health', 'GET', undefined, 20000)
    return r.status === 'ok'
  } catch (e) {
    return false
  }
}

module.exports = {
  checkArticle,
  uploadScreenshots,
  healthCheck
}
