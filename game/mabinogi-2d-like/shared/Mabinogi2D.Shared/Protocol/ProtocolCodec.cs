using MessagePack;

namespace Mabinogi2D.Shared.Protocol;

/// <summary>
/// 메시지를 Envelope로 감싸 바이트로 인코딩/디코딩하는 단일 진입점.
/// 서버와 클라이언트가 동일 코드를 공유하므로 직렬화 불일치가 생기지 않는다.
/// </summary>
public static class ProtocolCodec
{
    public static byte[] Encode<T>(MessageType type, T payload)
    {
        var env = new Envelope
        {
            Type = type,
            Payload = MessagePackSerializer.Serialize(payload),
        };
        return MessagePackSerializer.Serialize(env);
    }

    public static Envelope DecodeEnvelope(byte[] data)
        => MessagePackSerializer.Deserialize<Envelope>(data);

    public static T DecodePayload<T>(Envelope env)
        => MessagePackSerializer.Deserialize<T>(env.Payload);
}
