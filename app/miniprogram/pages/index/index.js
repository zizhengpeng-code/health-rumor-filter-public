// pages/index/index.js
const { healthCheck } = require('../../utils/api.js')

Page({
  data: {
    mode: 'url',          // 'url' | 'text' | 'image'
    url: '',
    images: [],           // 截图模式：本地临时路径（D54）
    text: '',
    title: '',
    checking: false,
    serverOnline: null    // null = 未检测, 'starting' = 冷启动中, true/false
  },

  onLoad() {
    this.checkServer()
  },

  onShow() {
    this.checkServer()
  },

  // 云托管没人用时会休眠（最小实例数为 0），唤醒要几十秒：先提示「正在启动」并重试，最多约 1 分钟才报连不上
  async checkServer() {
    if (this._checking) return
    this._checking = true
    try {
      for (let i = 0; i < 6; i++) {
        if (await healthCheck()) {
          this.setData({ serverOnline: true })
          return
        }
        this.setData({ serverOnline: 'starting' })
        await new Promise(r => setTimeout(r, 8000))
      }
      this.setData({ serverOnline: false })
    } finally {
      this._checking = false
    }
  },

  switchMode(e) {
    this.setData({ mode: e.currentTarget.dataset.mode })
  },

  onUrlInput(e) {
    this.setData({ url: e.detail.value })
  },

  onTextInput(e) {
    this.setData({ text: e.detail.value })
  },

  onTitleInput(e) {
    this.setData({ title: e.detail.value })
  },

  chooseImages() {
    const left = 3 - this.data.images.length
    if (left <= 0) return
    wx.chooseMedia({
      count: left,
      mediaType: ['image'],
      sizeType: ['compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const paths = res.tempFiles.map(f => f.tempFilePath)
        this.setData({ images: this.data.images.concat(paths).slice(0, 3) })
      }
    })
  },

  removeImage(e) {
    const images = this.data.images.slice()
    images.splice(e.currentTarget.dataset.index, 1)
    this.setData({ images })
  },

  previewImage(e) {
    wx.previewImage({ urls: this.data.images, current: this.data.images[e.currentTarget.dataset.index] })
  },

  async startCheck() {
    const { mode, url, text, title, images } = this.data

    if (mode === 'image') {
      if (!images.length) {
        wx.showToast({ title: '请先添加截图', icon: 'none' })
        return
      }
      if (getApp().globalData.backend !== 'cloud') {
        wx.showToast({ title: '截图核查需使用云端服务', icon: 'none' })
        return
      }
    } else if (mode === 'url') {
      if (!url.trim()) {
        wx.showToast({ title: '请输入文章链接', icon: 'none' })
        return
      }
      if (!url.includes('http')) {
        wx.showToast({ title: '请输入完整链接（含 http）', icon: 'none' })
        return
      }
    } else {
      if (text.trim().length < 50) {
        wx.showToast({ title: '正文太短，请粘贴完整内容', icon: 'none' })
        return
      }
    }

    this.setData({ checking: true })
    wx.showLoading({ title: '正在核查中……', mask: true })

    try {
      const { checkArticle, uploadScreenshots } = require('../../utils/api.js')
      let fileIDs = []
      if (mode === 'image') {
        wx.showLoading({ title: '正在上传截图……', mask: true })
        fileIDs = await uploadScreenshots(images)
        wx.showLoading({ title: '正在识别并核查……', mask: true })
      }
      const result = await checkArticle({
        url: mode === 'url' ? url.trim() : '',
        text: mode === 'text' ? text.trim() : '',
        title: mode === 'url' ? '' : title.trim(),
        images: fileIDs
      })

      wx.hideLoading()

      if (!result.ok) {
        wx.showModal({ title: '核查失败', content: result.error, showCancel: false })
        return
      }

      // 跳转到结果页
      wx.navigateTo({
        url: '/pages/result/result?data=' + encodeURIComponent(JSON.stringify(result))
      })
    } catch (err) {
      wx.hideLoading()
      wx.showModal({
        title: '出错了',
        content: err.message || '核查失败，请重试',
        showCancel: false
      })
    } finally {
      this.setData({ checking: false })
    }
  },

  goToAbout() {
    wx.navigateTo({ url: '/pages/about/about' })
  },

  onShareAppMessage() {
    return {
      title: '家庭群健康谣言核查 —— 帮爸妈查一查',
      path: '/pages/index/index'
    }
  }
})
