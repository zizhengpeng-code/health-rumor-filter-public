// app.js
App({
  globalData: {
    // 后端模式：
    //   'cloud' 微信云托管（正式使用：体验版 / 上线）
    //   'local' 本机后端（开发调试：先在电脑上启动 uvicorn）
    backend: 'local',

    // 云托管：环境 ID 在控制台右上角「环境」下拉里；服务名是云托管里新建的服务名
    cloudEnv: '',
    cloudService: '',

    // 本机后端地址（backend: 'local' 时使用）
    apiBase: 'http://localhost:8000'
  },

  onLaunch() {
    if (this.globalData.backend === 'cloud') {
      wx.cloud.init()
    }
    // 获取系统信息，用于适配
    const sys = wx.getSystemInfoSync()
    this.globalData.statusBarHeight = sys.statusBarHeight
    this.globalData.windowHeight = sys.windowHeight
    this.globalData.windowWidth = sys.windowWidth
  }
})
