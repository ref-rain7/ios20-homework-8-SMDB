/// Copyright (c) 2020 Razeware LLC
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import NaturalLanguage
import CoreML

func getLanguage(text: String) -> NLLanguage? {
    return NLLanguageRecognizer.dominantLanguage(for: text)
}

func getPeopleNames(text: String, block: (String) -> Void) {
    let tagger = NLTagger(tagSchemes: [.nameType])
    let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther, .joinNames]
    tagger.string = text
    tagger.enumerateTags(
        in: text.startIndex..<text.endIndex,
        unit: .word,
        scheme: .nameType,
        options: options,
        using: { tag, range in
            if tag == .personalName {
                block(String(text[range]))
            }
            return true
        }
    )
}

func getSearchTerms(text: String, language: String? = nil, block: (String) -> Void) {
    let tagger = NLTagger(tagSchemes: [.lemma])
    let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther, .joinNames]
    tagger.string = text
//    if let lan = language {
//        tagger.setLanguage(NLLanguage(rawValue: lan), range: text.startIndex..<text.endIndex)
//    }
    tagger.enumerateTags(
        in: text.startIndex..<text.endIndex,
        unit: .word,
        scheme: .lemma,
        options: options,
        using: { tag, range in
            let token = String(text[range]).lowercased()
            if let tag = tag {
                let lemma = tag.rawValue.lowercased()
                block(lemma)
                if lemma != token {
                    block(token)
                }
            } else {
                block(token)
            }
            return true
        }
    )
}

func analyzeSentiment(text:String) -> Double? {
    let tagger = NLTagger(tagSchemes: [.sentimentScore])
    let (tag, _) = tagger.tag(at: text.startIndex, unit: .document, scheme: .sentimentScore)
    guard let sentiment = tag, let score = Double(sentiment.rawValue) else {
        return nil
    }
    return score
}

func getSentimentClassifier() -> NLModel? {
    try! NLModel(mlModel: SentimentClassifier(configuration: MLModelConfiguration()).model)
}

func predictSentiment(text: String, sentimentClassifier: NLModel) -> String? {
    sentimentClassifier.predictedLabel(for: text)
}

// ------------------------------------------------------------------
// -------  Everything below here is for translation chapters -------
// ------------------------------------------------------------------

fileprivate let esChr2int = loadCharToIntJsonMap(from: "esCharToInt")
fileprivate let int2enChr = loadIntToCharJsonMap(from: "intToEnChar")
fileprivate let startTokenIndex = 0
fileprivate let stopTokenIndex = 1
fileprivate let maxOutSequenceLength = 60


fileprivate func getEncoderInput(text: String) -> MLMultiArray? {
    let cleanedText = text.filter { esChr2int.keys.contains($0) }
    if cleanedText.isEmpty {
        return nil
    }
    let vocabSize = esChr2int.count
    let encoderIn = initMultiArray(shape: [NSNumber(value: cleanedText.count), 1, NSNumber(value: vocabSize)])
    for (i, c) in cleanedText.enumerated() {
        encoderIn[i * vocabSize + esChr2int[c]!] = 1
    }
    return encoderIn
}

fileprivate func getDecoderInput(encoderInput: MLMultiArray) -> Es2EnCharDecoder16BitInput {
    let encoder = try! Es2EnCharEncoder16Bit(configuration: MLModelConfiguration())
    let encoderOut = try! encoder.prediction(
        encodedSeq: encoderInput,
        encoder_lstm_h_in: nil,
        encoder_lstm_c_in: nil
    )
    let decoderIn = initMultiArray(shape: [NSNumber(value: int2enChr.count)])
    return Es2EnCharDecoder16BitInput(
        encodedChar: decoderIn,
        decoder_lstm_h_in: encoderOut.encoder_lstm_h_out,
        decoder_lstm_c_in: encoderOut.encoder_lstm_c_out
    )
}

func getSentences(text: String) -> [String] {
    var sentences: [String] = []
    let tokenizer = NLTokenizer(unit: .sentence)
    tokenizer.string = text
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
        sentences.append(String(text[range]))
        return true
    }
    return sentences
}

func spanishToEnglish(text: String) -> String? {
    guard let encoderIn = getEncoderInput(text: text) else {
        return nil
    }
    var translatedText: [Character] = []
    let decoder = try! Es2EnCharDecoder16Bit(configuration: MLModelConfiguration())
    let decoderIn = getDecoderInput(encoderInput: encoderIn)
    var decodedIndex = startTokenIndex
    for _ in 0..<maxOutSequenceLength {
        decoderIn.encodedChar[decodedIndex] = 1
        let decoderOut = try! decoder.prediction(input: decoderIn)
        decoderIn.encodedChar[decodedIndex] = 0
        decodedIndex =  argmax(array: decoderOut.nextCharProbs)
        if decodedIndex == stopTokenIndex {
            break
        }
        translatedText.append(int2enChr[decodedIndex]!)
        decoderIn.decoder_lstm_c_in = decoderOut.decoder_lstm_c_out
        decoderIn.decoder_lstm_h_in = decoderOut.decoder_lstm_h_out
    }
    return String(translatedText)
}
