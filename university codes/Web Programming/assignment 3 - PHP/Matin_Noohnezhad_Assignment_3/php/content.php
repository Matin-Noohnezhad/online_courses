<?php
// define variables and set to empty values
$id = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $id = test_input($_POST["idNo"]);
    //
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "assignment_3";
    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    mysqli_set_charset($conn, "utf8");
// Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $sql = "SELECT title,content FROM data where id=" . $id;
    $result = $conn->query($sql);

    $contentTitle = "";
    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            $contentTitle .= "
<div>
            <h2>" . $row['title'] . "</h2>
            <p>" . $row['content'] . "</p>
      </div>      ";
        }
    }

    $string = "<!DOCTYPE html>
<html lang=\"fa\" dir='rtl'>
<head>
    <meta charset=\"UTF-8\">
    <title>Title</title>
</head>
<body>
" . $contentTitle . "
</body>
</html>";

    echo $string;


}
function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>