/*
 * Copyright 2025 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mlkit.genai.demo.kotlin

import android.app.AlertDialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.text.TextUtils
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.concurrent.futures.CallbackToFutureAdapter
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.google.common.util.concurrent.ListenableFuture
import com.google.mlkit.genai.common.DownloadCallback
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.StreamingCallback
import com.google.mlkit.genai.demo.ContentItem
import com.google.mlkit.genai.demo.GenerationConfigDialog
import com.google.mlkit.genai.demo.GenerationConfigUtils
import com.google.mlkit.genai.demo.R
import com.google.mlkit.genai.prompt.Candidate
import com.google.mlkit.genai.prompt.Candidate.FinishReason
import com.google.mlkit.genai.prompt.CountTokensResponse
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.GenerateContentResponse
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.PromptPrefix
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.prompt.createCachedContextRequest
import com.google.mlkit.genai.prompt.generateContentRequest
import java.io.IOException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.guava.future
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * An activity that demonstrates a chat-like interface for the Open Prompt API, allowing requests
 * with both text and images, and including generation configuration.
 */
class OpenPromptActivity :
  BaseActivity<ContentItem>(), GenerationConfigDialog.OnConfigUpdateListener {
  private val TAG = "OpenPromptActivity"
  private val ACTION_CLEAR_CACHES = 1000

  private var generativeModel: GenerativeModel? = null
  private lateinit var requestEditText: EditText
  private lateinit var sendButton: Button
  private lateinit var selectImageButton: ImageButton
  private lateinit var imagePreview: ImageView
  private lateinit var configButton: Button
  private lateinit var prefixEditText: EditText
  private lateinit var createCacheCheckBox: CheckBox

  private var selectedImageUri: Uri? = null

  private var curTemperature: Float? = null
  private var curTopK: Int? = null
  private var curSeed: Int? = null
  private var curMaxOutputTokens: Int? = null
  private var curCandidateCount: Int? = null
  private var useDefaultConfig = false
  private var useExplicitCache = false

  private val pickImageLauncher =
    registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
      if (uri != null) {
        selectedImageUri = uri
        Glide.with(this).load(uri).into(imagePreview)
        imagePreview.visibility = View.VISIBLE
        Toast.makeText(this, "1 image selected", Toast.LENGTH_SHORT).show()
      } else {
        Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    requestEditText = findViewById(R.id.request_edit_text)
    sendButton = findViewById(R.id.send_button)
    selectImageButton = findViewById(R.id.select_image_prompt_button)
    imagePreview = findViewById(R.id.image_thumbnail_preview_input)
    configButton = findViewById(R.id.config_button)
    prefixEditText = findViewById(R.id.prefix_edit_text)
    createCacheCheckBox = findViewById(R.id.create_cache_checkbox)
    createCacheCheckBox.setOnCheckedChangeListener { _, _ ->
      prefixEditText.setText("")
      updateRequestEditTextHint()
      updatePrefixEditTextState()
    }

    selectImageButton.setOnClickListener { pickImageLauncher.launch("image/*") }

    // Remove the selected image when the user clicks on the image preview.
    imagePreview.setOnClickListener {
      selectedImageUri = null
      imagePreview.visibility = View.GONE
      Glide.with(this).clear(imagePreview)
      Toast.makeText(this, "Image removed", Toast.LENGTH_SHORT).show()
    }

    configButton.setOnClickListener { GenerationConfigDialog().show(supportFragmentManager, null) }

    sendButton.setOnClickListener {
      if (useExplicitCache) {
        val cacheName = prefixEditText.text.toString().trim()
        if (TextUtils.isEmpty(cacheName)) {
          Toast.makeText(this, R.string.cache_name_empty, Toast.LENGTH_SHORT).show()
          return@setOnClickListener
        }
        val text = requestEditText.text.toString().trim()
        if (createCacheCheckBox.isChecked) {
          if (TextUtils.isEmpty(text)) {
            Toast.makeText(this, R.string.prefix_to_cache_empty, Toast.LENGTH_SHORT).show()
            return@setOnClickListener
          }
          onSend(ContentItem.CacheRequestItem.fromRequest(cacheName, text))
        } else {
          if (TextUtils.isEmpty(text)) {
            Toast.makeText(this, R.string.input_message_is_empty, Toast.LENGTH_SHORT).show()
            return@setOnClickListener
          }
          onSend(ContentItem.TextWithPrefixCacheItem.fromRequest(cacheName, text))
        }
        requestEditText.setText("")
        return@setOnClickListener
      }

      val requestText = requestEditText.text.toString().trim()
      if (TextUtils.isEmpty(requestText)) {
        Toast.makeText(this, R.string.input_message_is_empty, Toast.LENGTH_SHORT).show()
        return@setOnClickListener
      }

      val prefixText = prefixEditText.text.toString().trim()
      if (!TextUtils.isEmpty(prefixText) && selectedImageUri != null) {
        Toast.makeText(this, R.string.warning_prefix_used_with_image, Toast.LENGTH_LONG).show()
        return@setOnClickListener
      }

      val requestItem: ContentItem =
        if (selectedImageUri != null) {
          ContentItem.TextAndImagesItem.fromRequest(requestText, arrayListOf(selectedImageUri!!))
        } else if (!TextUtils.isEmpty(prefixText)) {
          ContentItem.TextWithPromptPrefixItem.fromRequest(prefixText, requestText)
        } else {
          ContentItem.TextItem.fromRequest(requestText)
        }
      onSend(requestItem)

      requestEditText.setText("")
      imagePreview.visibility = View.GONE
      Glide.with(this).clear(imagePreview)
      selectedImageUri = null
    }

    onConfigUpdated()

    initGenerator()
  }

  override fun onConfigUpdated() {
    useDefaultConfig = GenerationConfigUtils.getUseDefaultConfig(applicationContext)
    if (useDefaultConfig) {
      // Cache cannot be used in the simple utility API.
      GenerationConfigUtils.setUseExplicitCache(applicationContext, false)
    }
    useExplicitCache = GenerationConfigUtils.getUseExplicitCache(applicationContext)

    if (useExplicitCache) {
      prefixEditText.visibility = View.VISIBLE
      prefixEditText.setHint(R.string.hint_add_cache_name)

      createCacheCheckBox.visibility = View.VISIBLE
      configButton.visibility = View.VISIBLE
      selectImageButton.visibility = View.GONE
      imagePreview.visibility = View.GONE
      selectedImageUri = null
    } else {
      prefixEditText.visibility = if (useDefaultConfig) View.GONE else View.VISIBLE
      prefixEditText.setHint(R.string.hint_add_prompt_prefix)
      createCacheCheckBox.visibility = View.GONE
      configButton.visibility = if (useDefaultConfig) View.GONE else View.VISIBLE
      selectImageButton.visibility = if (useDefaultConfig) View.GONE else View.VISIBLE
      imagePreview.visibility =
        if (useDefaultConfig || selectedImageUri == null) View.GONE else View.VISIBLE
    }
    prefixEditText.setText("")
    requestEditText.setText("")
    updateRequestEditTextHint()
    updatePrefixEditTextState()

    curTemperature = GenerationConfigUtils.getTemperature(applicationContext)
    curTopK = GenerationConfigUtils.getTopK(applicationContext)
    curSeed = GenerationConfigUtils.getSeed(applicationContext)
    curCandidateCount = GenerationConfigUtils.getCandidateCount(applicationContext)
    curMaxOutputTokens = GenerationConfigUtils.getMaxOutputTokens(applicationContext)
  }

  private fun updateRequestEditTextHint() {
    requestEditText.setHint(
      if (useExplicitCache) {
        if (createCacheCheckBox.isChecked) {
          R.string.hint_add_prefix_to_cache
        } else {
          R.string.hint_add_suffix_for_inference
        }
      } else {
        R.string.hint_type_a_message
      }
    )
  }

  override fun getLayoutResId(): Int = R.layout.activity_openprompt

  override fun getBaseModelName(): ListenableFuture<String> =
    lifecycleScope.future { checkNotNull(generativeModel).getBaseModelName() }

  override fun checkFeatureStatus(): ListenableFuture<Int> =
    lifecycleScope.future { checkNotNull(generativeModel).checkStatus() }

  override fun downloadFeature(callback: DownloadCallback): ListenableFuture<Void> {
    return CallbackToFutureAdapter.getFuture { completer ->
      val job =
        lifecycleScope.launch {
          try {
            checkNotNull(generativeModel).download().collect { status ->
              when (status) {
                is DownloadStatus.DownloadStarted ->
                  callback.onDownloadStarted(status.bytesToDownload)
                is DownloadStatus.DownloadProgress ->
                  callback.onDownloadProgress(status.totalBytesDownloaded)
                is DownloadStatus.DownloadFailed -> callback.onDownloadFailed(status.e)
                is DownloadStatus.DownloadCompleted -> callback.onDownloadCompleted()
              }
            }
            completer.set(null)
          } catch (e: Exception) {
            completer.setException(e)
          }
        }

      completer.addCancellationListener({ job.cancel() }, ContextCompat.getMainExecutor(this))

      "downloadFeature"
    }
  }

  override fun runInferenceImpl(
    request: ContentItem,
    streamingCallback: StreamingCallback?,
  ): ListenableFuture<List<String>> {
    if (request is ContentItem.CacheRequestItem) {
      return lifecycleScope.future { listOf(createCache(request)) }
    }
    return lifecycleScope.future {
      if (request is ContentItem.TextItem && useDefaultConfig) {
        // useDefaultConfig is used for the case where user wants to use utility function with
        // default config values
        val result =
          if (streamingCallback != null) {
            checkNotNull(generativeModel).generateContent(request.text, streamingCallback)
          } else {
            checkNotNull(generativeModel).generateContent(request.text)
          }
        return@future resultToStrings(result)
      }

      val genRequest = createGenerateContentRequest(request)
      val result =
        if (streamingCallback != null) {
          checkNotNull(generativeModel).generateContent(genRequest, streamingCallback)
        } else {
          checkNotNull(generativeModel).generateContent(genRequest)
        }
      resultToStrings(result)
    }
  }

  override fun runInferenceStreamImpl(request: ContentItem): Flow<String>? {
    if (request is ContentItem.CacheRequestItem) {
      return flow { emit(createCache(request)) }
    }
    if (request is ContentItem.TextItem && useDefaultConfig) {
      // useDefaultConfig is used for the case where user wants to use utility function with
      // default config values
      return flow {
        checkNotNull(generativeModel)
          .generateContentStream(request.text)
          .map { result ->
            val text = result.candidates.first().text
            val finishReason = result.candidates.first().finishReason
            if (finishReason == Candidate.FinishReason.MAX_TOKENS) {
              "$text\n(FinishReason: MAX_TOKENS)"
            } else {
              text
            }
          }
          .collect { emit(it) }
      }
    }
    return flow {
      val genRequest = createGenerateContentRequest(request)
      checkNotNull(generativeModel)
        .generateContentStream(genRequest)
        .map { result ->
          val text = result.candidates.first().text
          val finishReason = result.candidates.first().finishReason
          if (finishReason == Candidate.FinishReason.MAX_TOKENS) {
            "$text\n(FinishReason: MAX_TOKENS)"
          } else {
            text
          }
        }
        .collect { emit(it) }
    }
  }

  private fun showCacheSelectionDialog() {
    lifecycleScope.launch {
      val caches = checkNotNull(generativeModel).caches.list()
      if (caches.isEmpty()) {
        Toast.makeText(this@OpenPromptActivity, "No caches available to select", Toast.LENGTH_SHORT)
          .show()
        return@launch
      }
      val cacheNames = caches.map { it.name }.toTypedArray()
      AlertDialog.Builder(this@OpenPromptActivity)
        .setTitle("Select Cache")
        .setItems(cacheNames) { _, which -> prefixEditText.setText(cacheNames[which]) }
        .show()
    }
  }

  private fun updatePrefixEditTextState() {
    if (useExplicitCache && !createCacheCheckBox.isChecked) {
      prefixEditText.isFocusable = false
      prefixEditText.isClickable = true
      prefixEditText.setOnClickListener { showCacheSelectionDialog() }
      prefixEditText.setHint(R.string.hint_select_cache_name)
    } else {
      prefixEditText.isFocusable = true
      prefixEditText.isFocusableInTouchMode = true
      prefixEditText.isClickable = false
      prefixEditText.setOnClickListener(null)
      if (useExplicitCache) {
        prefixEditText.setHint(R.string.hint_add_cache_name)
      } else {
        prefixEditText.setHint(R.string.hint_add_prompt_prefix)
      }
    }
  }

  private suspend fun createCache(request: ContentItem.CacheRequestItem): String {
    val unused =
      checkNotNull(generativeModel)
        .caches
        .create(createCachedContextRequest(request.cacheName, PromptPrefix(request.prefixToCache)))

    // Return a string to indicate the cache is created successfully.
    return "${getString(R.string.prefix_cached)}: ${request.cacheName}"
  }

  private fun createGenerateContentRequest(request: ContentItem): GenerateContentRequest {
    var requestText = ""
    var promptPrefixText = ""
    var imageBitmap: Bitmap? = null
    var cachedContextNameText: String? = null

    when (request) {
      is ContentItem.TextItem -> {
        requestText = request.text
      }
      is ContentItem.TextAndImagesItem -> {
        requestText = request.text
        for (uri in request.imageUris) {
          try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
              BitmapFactory.decodeStream(inputStream)?.let { bitmap -> imageBitmap = bitmap }
            }
          } catch (e: IOException) {
            Log.e(TAG, "Error decoding image URI: $uri", e)
          }
        }
      }
      is ContentItem.ImageItem -> {
        try {
          contentResolver.openInputStream(request.imageUri)?.use { inputStream ->
            BitmapFactory.decodeStream(inputStream)?.let { bitmap -> imageBitmap = bitmap }
          }
        } catch (e: IOException) {
          Log.e(TAG, "Error decoding image URI: ${request.imageUri}", e)
        }
      }
      is ContentItem.TextWithPromptPrefixItem -> {
        requestText = request.dynamicSuffix
        promptPrefixText = request.promptPrefix
      }
      is ContentItem.TextWithPrefixCacheItem -> {
        requestText = request.dynamicSuffix
        cachedContextNameText = request.cacheName
      }
      is ContentItem.CacheRequestItem -> {
        throw IllegalStateException("CacheRequestItem is for creating cache only.")
      }
    }

    return if (imageBitmap != null) {
      generateContentRequest(ImagePart(imageBitmap), TextPart(requestText)) {
        temperature = curTemperature
        topK = curTopK
        seed = curSeed
        maxOutputTokens = curMaxOutputTokens
        candidateCount = curCandidateCount
      }
    } else {
      generateContentRequest(TextPart(requestText)) {
        if (useExplicitCache) {
          cachedContextName = cachedContextNameText
        } else {
          promptPrefix = PromptPrefix(promptPrefixText)
        }
        temperature = curTemperature
        topK = curTopK
        seed = curSeed
        maxOutputTokens = curMaxOutputTokens
        candidateCount = curCandidateCount
      }
    }
  }

  private fun resultToStrings(result: GenerateContentResponse): List<String> =
    result.candidates.map { candidate ->
      val text = candidate.text
      if (candidate.finishReason == Candidate.FinishReason.MAX_TOKENS) {
        "$text\n(FinishReason: MAX_TOKENS)"
      } else {
        text
      }
    }

  override fun runInferenceForBatchTask(request: String): List<String> {
    return runBlocking {
      val resultText =
        try {
          if (useDefaultConfig) {
            // useDefaultConfig is used for the case where user wants to use utility function with
            // default config values
            checkNotNull(generativeModel).generateContent(request).candidates.first().text
          } else {
            val genRequest =
              generateContentRequest(TextPart(request)) {
                temperature = curTemperature
                topK = curTopK
                seed = curSeed
                maxOutputTokens = curMaxOutputTokens
                candidateCount = curCandidateCount
              }
            checkNotNull(generativeModel).generateContent(genRequest).candidates.first().text
          }
        } catch (e: Exception) {
          "Failed to run inference: ${e.message}"
        }
      listOf(checkNotNull(resultText))
    }
  }

  override suspend fun countTokens(request: ContentItem): CountTokensResponse {
    if (request is ContentItem.CacheRequestItem) {
      // Count tokens does not support for cache request by now.
      return CountTokensResponse(0)
    }
    val genRequest = createGenerateContentRequest(request)
    return checkNotNull(generativeModel).countTokens(genRequest)
  }

  override suspend fun getTokenLimit(): Int {
    return checkNotNull(generativeModel).getTokenLimit()
  }

  override fun startGeneratingUi() {
    super.startGeneratingUi()
    sendButton.isEnabled = false
    requestEditText.isEnabled = false
    selectImageButton.isEnabled = false
    sendButton.setText(R.string.generating)
  }

  override fun endGeneratingUi(debugInfo: String) {
    super.endGeneratingUi(debugInfo)
    sendButton.isEnabled = true
    requestEditText.isEnabled = true
    selectImageButton.isEnabled = true
    sendButton.setText(R.string.button_send)
  }

  private fun initGenerator() {
    generativeModel?.close()
    generativeModel = com.google.mlkit.genai.prompt.Generation.getClient()
    resetProcessor()
  }

  override fun onCreateOptionsMenu(menu: Menu): Boolean {
    if (!super.onCreateOptionsMenu(menu)) {
      return false
    }
    menu.add(Menu.NONE, ACTION_CLEAR_CACHES, Menu.NONE, "Clear all prefix caches")
    return true
  }

  override fun onPrepareOptionsMenu(menu: Menu): Boolean {
    menu.findItem(R.id.action_simple_api)?.apply {
      isVisible = true
      isChecked = useDefaultConfig
    }
    menu.findItem(R.id.action_explicit_cache)?.apply {
      isVisible = true
      isChecked = useExplicitCache
      isEnabled = !useDefaultConfig
    }
    return super.onPrepareOptionsMenu(menu)
  }

  override fun onOptionsItemSelected(item: MenuItem): Boolean {
    when (item.itemId) {
      ACTION_CLEAR_CACHES -> {
        lifecycleScope.launch {
          if (useExplicitCache) {
            val caches = checkNotNull(generativeModel).caches.list()
            if (caches.isNotEmpty()) {
              Log.d(TAG, "Going to delete explicit caches, size: ${caches.size}")
              for (cacheName in caches.map { it.name }) {
                if (checkNotNull(generativeModel).caches.delete(cacheName)) {
                  Log.d(TAG, "Deleted explicit cache: $cacheName")
                } else {
                  Log.d(TAG, "Failed to delete explicit cache: $cacheName")
                }
              }
              prefixEditText.setText("")
            }
          } else {
            checkNotNull(generativeModel).clearImplicitCaches()
            Log.d(TAG, "Cleared implicit caches")
          }
          Toast.makeText(this@OpenPromptActivity, "Caches cleared", Toast.LENGTH_SHORT).show()
        }
        return true
      }
      R.id.action_simple_api -> {
        val newState = !item.isChecked
        item.isChecked = newState
        GenerationConfigUtils.setUseDefaultConfig(applicationContext, newState)
        onConfigUpdated()
        return true
      }
      R.id.action_explicit_cache -> {
        val newState = !item.isChecked
        item.isChecked = newState
        GenerationConfigUtils.setUseExplicitCache(applicationContext, newState)
        onConfigUpdated()
        return true
      }
    }
    return super.onOptionsItemSelected(item)
  }
}
