// On-device OCR using macOS Vision. No network calls.
import Foundation
import Vision

do {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = CommandLine.arguments.contains("--fast") ? .fast : .accurate
    request.usesCPUOnly = true
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.minimumTextHeight = 0.008
    let handler = VNImageRequestHandler(data: input, options: [:])
    try handler.perform([request])
    var rows: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let b = observation.boundingBox
        rows.append(["text": candidate.string, "confidence": candidate.confidence,
                     "box": [b.minX, 1 - b.maxY, b.maxX, 1 - b.minY]])
    }
    let output = try JSONSerialization.data(withJSONObject: rows)
    FileHandle.standardOutput.write(output)
} catch {
    FileHandle.standardError.write(Data("Local text recognition failed: \(error)\n".utf8))
    exit(1)
}
