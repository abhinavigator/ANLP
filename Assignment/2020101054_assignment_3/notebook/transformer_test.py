{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":30558,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import numpy as np\nimport torch\nimport math\nfrom torch import nn\nimport torch.nn.functional as F\n\ndef get_device():\n    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')\n\ndef scaled_dot_product(q, k, v, mask=None):\n    d_k = q.size()[-1]\n    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)\n    if mask is not None:\n        scaled = scaled.permute(1, 0, 2, 3) + mask\n        scaled = scaled.permute(1, 0, 2, 3)\n    attention = F.softmax(scaled, dim=-1)\n    values = torch.matmul(attention, v)\n    return values, attention\n\nclass PositionalEncoding(nn.Module):\n    def __init__(self, d_model, max_sequence_length):\n        super().__init__()\n        self.max_sequence_length = max_sequence_length\n        self.d_model = d_model\n\n    def forward(self):\n        even_i = torch.arange(0, self.d_model, 2).float()\n        denominator = torch.pow(10000, even_i/self.d_model)\n        position = (torch.arange(self.max_sequence_length)\n                          .reshape(self.max_sequence_length, 1))\n        even_PE = torch.sin(position / denominator)\n        odd_PE = torch.cos(position / denominator)\n        stacked = torch.stack([even_PE, odd_PE], dim=2)\n        PE = torch.flatten(stacked, start_dim=1, end_dim=2)\n        return PE\n\nclass SentenceEmbedding(nn.Module):\n    \"For a given sentence, create an embedding\"\n    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):\n        super().__init__()\n        self.vocab_size = len(language_to_index)\n        self.max_sequence_length = max_sequence_length\n        self.embedding = nn.Embedding(self.vocab_size, d_model)\n        self.language_to_index = language_to_index\n        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)\n        self.dropout = nn.Dropout(p=0.1)\n        self.START_TOKEN = START_TOKEN\n        self.END_TOKEN = END_TOKEN\n        self.PADDING_TOKEN = PADDING_TOKEN\n    \n    def batch_tokenize(self, batch, start_token, end_token):\n\n#         def tokenize(sentence, start_token, end_token):\n#             sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]\n#             if start_token:\n#                 sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])\n#             if end_token:\n#                 sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])\n#             for _ in range(len(sentence_word_indicies), self.max_sequence_length):\n#                 sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])\n#             return torch.tensor(sentence_word_indicies)\n        \n        def tokenize(sentence, start_token, end_token):\n            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence) if token in self.language_to_index]\n            if start_token:\n                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])\n            if end_token:\n                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])\n            return sentence_word_indicies\n\n\n        tokenized = []\n        for sentence_num in range(len(batch)):\n           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )\n        tokenized = torch.stack(tokenized)\n        return tokenized.to(get_device())\n    \n    def forward(self, x, start_token, end_token): # sentence\n        x = self.batch_tokenize(x, start_token, end_token)\n        x = self.embedding(x)\n        pos = self.position_encoder().to(get_device())\n        x = self.dropout(x + pos)\n        return x\n\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.num_heads = num_heads\n        self.head_dim = d_model // num_heads\n        self.qkv_layer = nn.Linear(d_model , 3 * d_model)\n        self.linear_layer = nn.Linear(d_model, d_model)\n    \n    def forward(self, x, mask):\n        batch_size, sequence_length, d_model = x.size()\n        qkv = self.qkv_layer(x)\n        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)\n        qkv = qkv.permute(0, 2, 1, 3)\n        q, k, v = qkv.chunk(3, dim=-1)\n        values, attention = scaled_dot_product(q, k, v, mask)\n        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)\n        out = self.linear_layer(values)\n        return out\n\n\nclass LayerNormalization(nn.Module):\n    def __init__(self, parameters_shape, eps=1e-5):\n        super().__init__()\n        self.parameters_shape=parameters_shape\n        self.eps=eps\n        self.gamma = nn.Parameter(torch.ones(parameters_shape))\n        self.beta =  nn.Parameter(torch.zeros(parameters_shape))\n\n    def forward(self, inputs):\n        dims = [-(i + 1) for i in range(len(self.parameters_shape))]\n        mean = inputs.mean(dim=dims, keepdim=True)\n        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)\n        std = (var + self.eps).sqrt()\n        y = (inputs - mean) / std\n        out = self.gamma * y + self.beta\n        return out\n\n  \nclass PositionwiseFeedForward(nn.Module):\n    def __init__(self, d_model, hidden, drop_prob=0.1):\n        super(PositionwiseFeedForward, self).__init__()\n        self.linear1 = nn.Linear(d_model, hidden)\n        self.linear2 = nn.Linear(hidden, d_model)\n        self.relu = nn.ReLU()\n        self.dropout = nn.Dropout(p=drop_prob)\n\n    def forward(self, x):\n        x = self.linear1(x)\n        x = self.relu(x)\n        x = self.dropout(x)\n        x = self.linear2(x)\n        return x\n\n\nclass EncoderLayer(nn.Module):\n    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):\n        super(EncoderLayer, self).__init__()\n        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)\n        self.norm1 = LayerNormalization(parameters_shape=[d_model])\n        self.dropout1 = nn.Dropout(p=drop_prob)\n        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)\n        self.norm2 = LayerNormalization(parameters_shape=[d_model])\n        self.dropout2 = nn.Dropout(p=drop_prob)\n\n    def forward(self, x, self_attention_mask):\n        residual_x = x.clone()\n        x = self.attention(x, mask=self_attention_mask)\n        x = self.dropout1(x)\n        x = self.norm1(x + residual_x)\n        residual_x = x.clone()\n        x = self.ffn(x)\n        x = self.dropout2(x)\n        x = self.norm2(x + residual_x)\n        return x\n    \nclass SequentialEncoder(nn.Sequential):\n    def forward(self, *inputs):\n        x, self_attention_mask  = inputs\n        for module in self._modules.values():\n            x = module(x, self_attention_mask)\n        return x\n\nclass Encoder(nn.Module):\n    def __init__(self, \n                 d_model, \n                 ffn_hidden, \n                 num_heads, \n                 drop_prob, \n                 num_layers,\n                 max_sequence_length,\n                 language_to_index,\n                 START_TOKEN,\n                 END_TOKEN, \n                 PADDING_TOKEN):\n        super().__init__()\n        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)\n        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)\n                                      for _ in range(num_layers)])\n\n    def forward(self, x, self_attention_mask, start_token, end_token):\n        x = self.sentence_embedding(x, start_token, end_token)\n        x = self.layers(x, self_attention_mask)\n        return x\n\n\nclass MultiHeadCrossAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.num_heads = num_heads\n        self.head_dim = d_model // num_heads\n        self.kv_layer = nn.Linear(d_model , 2 * d_model)\n        self.q_layer = nn.Linear(d_model , d_model)\n        self.linear_layer = nn.Linear(d_model, d_model)\n    \n    def forward(self, x, y, mask):\n        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention\n        kv = self.kv_layer(x)\n        q = self.q_layer(y)\n        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)\n        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)\n        kv = kv.permute(0, 2, 1, 3)\n        q = q.permute(0, 2, 1, 3)\n        k, v = kv.chunk(2, dim=-1)\n        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!\n        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)\n        out = self.linear_layer(values)\n        return out\n\n\nclass DecoderLayer(nn.Module):\n    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):\n        super(DecoderLayer, self).__init__()\n        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)\n        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])\n        self.dropout1 = nn.Dropout(p=drop_prob)\n\n        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)\n        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])\n        self.dropout2 = nn.Dropout(p=drop_prob)\n\n        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)\n        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])\n        self.dropout3 = nn.Dropout(p=drop_prob)\n\n    def forward(self, x, y, self_attention_mask, cross_attention_mask):\n        _y = y.clone()\n        y = self.self_attention(y, mask=self_attention_mask)\n        y = self.dropout1(y)\n        y = self.layer_norm1(y + _y)\n\n        _y = y.clone()\n        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)\n        y = self.dropout2(y)\n        y = self.layer_norm2(y + _y)\n\n        _y = y.clone()\n        y = self.ffn(y)\n        y = self.dropout3(y)\n        y = self.layer_norm3(y + _y)\n        return y\n\n\nclass SequentialDecoder(nn.Sequential):\n    def forward(self, *inputs):\n        x, y, self_attention_mask, cross_attention_mask = inputs\n        for module in self._modules.values():\n            y = module(x, y, self_attention_mask, cross_attention_mask)\n        return y\n\nclass Decoder(nn.Module):\n    def __init__(self, \n                 d_model, \n                 ffn_hidden, \n                 num_heads, \n                 drop_prob, \n                 num_layers,\n                 max_sequence_length,\n                 language_to_index,\n                 START_TOKEN,\n                 END_TOKEN, \n                 PADDING_TOKEN):\n        super().__init__()\n        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)\n        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])\n\n    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):\n        y = self.sentence_embedding(y, start_token, end_token)\n        y = self.layers(x, y, self_attention_mask, cross_attention_mask)\n        return y\n\n\nclass Transformer(nn.Module):\n    def __init__(self, \n                d_model, \n                ffn_hidden, \n                num_heads, \n                drop_prob, \n                num_layers,\n                max_sequence_length, \n                kn_vocab_size,\n                english_to_index,\n                kannada_to_index,\n                START_TOKEN, \n                END_TOKEN, \n                PADDING_TOKEN\n                ):\n        super().__init__()\n        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)\n        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)\n        self.linear = nn.Linear(d_model, kn_vocab_size)\n        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')\n\n    def forward(self, \n                x, \n                y, \n                encoder_self_attention_mask=None, \n                decoder_self_attention_mask=None, \n                decoder_cross_attention_mask=None,\n                enc_start_token=False,\n                enc_end_token=False,\n                dec_start_token=False, # We should make this true\n                dec_end_token=False): # x, y are batch of sentences\n        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)\n        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)\n        out = self.linear(out)\n        return out","metadata":{"_uuid":"25e12d71-f8de-46a0-bf23-18b95cc920c2","_cell_guid":"cdbf9836-c488-45a1-9ab9-5cd0ac0eac5d","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}